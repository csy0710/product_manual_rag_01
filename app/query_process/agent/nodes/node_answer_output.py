import sys
from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
import re
_IMAGE_BLOCK_MARKER = "【图片】"
MAX_CONTEXT_CHARS = 12000 #限制prompt最大长度
def step_1_check_answer(state):
    """
    检查state中是否有answer字段
    :param state:
    :return:
    """
    #获取answer字段的值||is_stream字段
    answer = state.get("answer",'')
    is_stream = state.get("is_stream",False)
    #判断answer字段是否存在为空
    if answer:
        #判断是不是流式
        if is_stream:
            push_to_session(state["session_id"], SSEEvent.DELTA,{"delta":answer})
        else:
        #非流式
            set_task_result(state["session_id"],{"answer":answer})
        return True
    else:
        return False


def step_2_load_prompt(state):
    """
    生成对应润色的提示词prompt
    :param state:
    :return:
    """
    #从state中获取数据
    rewritten_query = state.get("rewritten_query",'') or state.get("original_query")
    reranked_docs = state.get("reranked_docs",[])
    item_names = state.get("item_names",[])
    history = state.get("history",[])
    #先处理chunk块的内容
    docs = []
    used_length = 0
    #把数据处理成["[[1][text][source][url][title][score]\n\n]" "[[1][text][source][url][title][score]\n\n]"]
    for i,doc in enumerate(reranked_docs,start=1):
        text = doc.get("text"),
        source = doc.get("source"),
        title = doc.get("title"),
        score = doc.get("score"),
        content = f"[{i}][source={source}][title={title}][score={score}]\n\n{text}"
        if used_length + len(content) > MAX_CONTEXT_CHARS:
            logger.info(f"警告：文档内容超过最大长度{MAX_CONTEXT_CHARS}，已截断")
            break
        docs.append(content)
        used_length += (len(content))
    final_context = "\n\n".join(docs)
    #处理history聊天记录的内容
    history_str = ""  #存储对话记录的字符内容
    if history and len(history)>0:
        for i,message in enumerate(history,start=1):
            role = message.get("role")
            text = message.get("text")
            current_history = ""
            if role == "user" and text:
                current_history = f"{role}：{text}"
            elif role == "assistant" and text:
                current_history = f"{role}：{text}"
            history_str += current_history
            used_length += (len(current_history))
            if used_length > MAX_CONTEXT_CHARS:
                logger.info(f"警告：对话记录超过最大长度{MAX_CONTEXT_CHARS}，已截断")
                break
    else:
        history_str = "没有历史对话记录"
    #处理item_name
    item_names_str = ",".join(item_names)
    #处理question问题
    answer_out_prompt = load_prompt("answer_out",context=final_context,history=history_str,item_names=item_names_str,question=rewritten_query)
    logger.info(f"已经完成了提示词生成：{answer_out_prompt}")
    return answer_out_prompt


def step_3_create_answer(state, prompt):
    """
    使用模型润色答案-》结果
    :param state:
    :param prompt:
    :return:
    """
    #使用模型生成最终的答案
    model = get_llm_client()
    answer = ''
    #获取流式状态【sse|set_task_result】
    is_stream = state.get("is_stream",False)
    if is_stream:
        #流式
        for chunk in model.stream(prompt):
            delta = chunk.content
            answer += delta
            push_to_session(state["session_id"], SSEEvent.DELTA,{"delta":delta})
    else:
        #非流式最终的结果直接赋值给state
        # 调用模型生成sse stream || set_result invoke
        response = model.invoke(prompt)
        content = response.content
        answer = content
        set_task_result(state["session_id"],"answer",content)

    #最终的答案赋值给state中的answer字段
    #返回最终的答案
    state['answer'] = answer
    logger.info(f"已经完成了答案生成：{answer}")
    return answer


def step_4_extract_images_url(state):
    """
    从原来toplist中提取图片地址
    从local chunk text中提取图片地址
    从web url 中提取图片地址
    :param state:
    :return:
    """
    images = []#存储图片
    set_images = set()#图片重复判断

    #定义正则
    image_reg = re.compile(r"!\[.*?\]\((.*?)\)")
    #循环处理切片 从高分到低分
    reranked_docs = state.get("reranked_docs",[])
    for doc in reranked_docs:
        url = doc.get("url")
        #判断是不是图片
        if url:
            if url.endswith((".jpg",".png",".jpeg",".gif",".webp")):
                if url not in set_images:
                    images.append(url)
                    set_images.add(url)
        text = doc.get("text")
        if text:
            #从text中提取图片地址
            matches = image_reg.findall(text)
            for match in matches:
                if match not in set_images:
                    images.append(match)
                    set_images.add(match)
    logger.info(f"已经完成了图片地址提取：{images}。图片数量：{len(images)}")
    state["images_urls"] = images
    return images


def step_5_write_history(state):
    """
    记录对话的聊天记录（用户/助手）
    :param state:
    :return:
    """
    session_id = state.get("session_id")
    answer = state.get("answer")
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    item_names = state.get("item_names",[])


    if answer:
        #assistant
        save_chat_message(
            session_id = session_id,
            role = "assistant",
            text = answer,
            item_names = item_names,
            rewritten_query = rewritten_query
        )
    logger.info(f"已经完成了对话记录的存储：{answer}")





def node_answer_output(state) :
    """
    宏观：将最终topk给大模型经过大模型润色之后将答案返回给前端（如果是流式，通过sse交给前端，如果不是流式得到结果后使用set_task_result返回结果）
    1.先检查state中是否有answer字段，如果有answer字段，将answer字段的值返回给前端
    2.生成对应润色的提示词prompt
    3.使用模型润色答案-》结果
    4.提取原来toplist中的图片地址，单独返回sse
    5.对话的聊天记录（用户/助手）
    """
    logger.info(f">>> [Stub] 执行节点: {sys._getframe().f_code.co_name}")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    #先检查state中是否有answer字段，如果有answer字段，将answer字段的值返回给前端
    answer_exist = step_1_check_answer(state)
    if not answer_exist:
        #生成对应润色的提示词prompt
        prompt = step_2_load_prompt(state)
        #3.使用模型润色答案-》结果
        answer = step_3_create_answer(state,prompt)
        #4.提取原来toplist中的图片地址，单独返回sse
        images_url = step_4_extract_images_url(state)
        if images_url:
            #不管是否流式，都返回图片地址
            push_to_session(state["session_id"], SSEEvent.FINAL,{
                "answer":answer,
                "status":"completed",
                "images_urls":images_url
            })
    #数据推送完成
    #记录history
    step_5_write_history(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    return state


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_answer_output 本地测试")
    print("=" * 50)

    # 1. 构造模拟数据
    # 模拟重排序后的文档列表 (reranked_docs)
    # 包含：本地文档（带Markdown图片）、联网结果（带URL字段）、纯文本文档
    mock_reranked_docs = [
        {
            "chunk_id": "local_101",
            "source": "local",
            "title": "HAK 180 烫金机操作手册_v2.pdf",
            "score": 0.95,
            "text": """
            HAK 180 烫金机的操作面板位于机器正前方。
            开启电源后，您需要先设置温度，默认建议设置在 110℃ 左右。
            具体的操作面板布局请参考下图：
            ![操作面板布局图](http://www.baidu.com/img/bd_logo.png)

            如果是进行局部烫金，请调节侧面的旋钮。
            ![侧面旋钮细节](http://local-server/images/knob_detail.png)
            """
        },
        {
            "chunk_id": None,
            "source": "web",
            "title": "HAK 180 常见故障排除 - 官网",
            "score": 0.88,
            "url": "http://example.com/hak180_troubleshooting.jpeg",  # 这是一个直接指向图片的URL（虽然少见，但用于测试提取）
            "text": "如果机器无法加热，请检查保险丝是否熔断..."
        },
        {
            "chunk_id": "local_102",
            "source": "local",
            "title": "安全注意事项",
            "score": 0.82,
            "text": "操作时请务必佩戴隔热手套，避免高温烫伤。"
        }
    ]

    # 模拟历史记录
    mock_history = [
        {"role": "user", "text": "你好，这款机器怎么用？"},
        {"role": "assistant", "text": "您好！请问您具体指的是哪一款机器？"},
        {"role": "user", "text": "HAK 180 烫金机"}
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_answer_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤和面板设置方法",
        "item_names": ["HAK 180 烫金机"],
        "history": mock_history,
        "reranked_docs": mock_reranked_docs,
        "is_stream": False,  # 测试非流式
        # "is_stream": True, # 若要测试流式，需确保 SSE 环境或 mock 相关函数
        "answer": None  # 初始无答案
    }

    try:
        # 运行节点
        result = node_answer_output(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")

        # 1. 验证 Prompt 构建
        if "prompt" in result:
            print(f"[PASS] Prompt 构建成功 (长度: {len(result['prompt'])})")
            # print(f"Prompt 预览:\n{result['prompt'][:200]}...")
        else:
            print("[FAIL] Prompt 未构建")

        # 2. 验证答案生成
        answer = result.get("answer")
        if answer and len(answer) > 10:
            print(f"[PASS] 答案生成成功 (长度: {len(answer)})")
            print(f"答案预览: {answer[:50]}...")
        else:
            print(f"[WARN] 答案生成可能异常 (Content: {answer})")

        # 3. 验证图片提取
        # 我们期望提取到 3 张图片：
        # 1. http://local-server/images/panel_view.jpg (来自 local_101)
        # 2. http://local-server/images/knob_detail.png (来自 local_101)
        # 3. http://example.com/hak180_troubleshooting.jpeg (来自 web 结果的 url 字段)

        # 注意：这里我们没办法直接从 result state 里拿到 image_urls，因为它是作为 SSE 推送出去的，或者存库了
        # 但我们可以通过日志观察 _extract_images_from_docs 的输出
        # 如果需要验证，可以临时修改 node_answer_output 返回 image_urls
        print("\n[INFO] 请检查上方日志中是否包含 '图片提取完成' 及以下 URL:")
        print(" - http://local-server/images/panel_view.jpg")
        print(" - http://local-server/images/knob_detail.png")
        print(" - http://example.com/hak180_troubleshooting.jpeg")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")