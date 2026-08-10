import json
import time
import sys

from langchain_core.messages import HumanMessage

from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.clients.mongo_history_utils import save_chat_message
from app.clients.mongo_history_utils import get_recent_messages
from app.core.load_prompt import load_prompt
from app.lm.embedding_utils import generate_embeddings
from app.lm.lm_utils import get_llm_client
from app.utils.task_utils import add_running_task, add_done_task


def step_3_llm_item_name_and_rewrite_query(original_query, history_chats):
    """
    根据历史记录，识别item_names和重写问题
    :param original_query:
    :param history_chats:
    :return:rewrite_query,item_names
    """
    #获取历史记录
    history_text = ""
    for chat in history_chats:
        history_text +=f"聊天角色：{chat["role"]}，回答内容：{chat["text"]}，重写问题：{chat["rewritten_query"]}，关联主体：{'，'.join(chat.get("item_names",[]))}，时间：{chat["ts"]}"

    #准备提示词
    prompt = load_prompt("rewritten_query_and_itemnames",history_text=history_text,query=original_query)


    lm_client = get_llm_client(json_mode=True)
    messages = [
        HumanMessage(content=prompt)
    ]
    response = lm_client.invoke(messages)
    #结果解析
    content = response.content

    if content.startswith("```json"):
        content = content.replace("```json","").replace("```","")
    dict_content = json.loads(content)

    if "item_names" not in dict_content:
        dict_content["item_names"] = []
    if "rewritten_query" not in dict_content:
        dict_content["rewritten_query"] = original_query

    logger.info(f"item_names: {dict_content['item_names']}")

    return dict_content


def step_4_query_milvus_item_names(item_names):
    """
    根据item_names，在向量库中搜索
    :param item_names:
    :return: final_result.append({
            "extracted":item_name,
            "matches":matches
        })
    """
    final_result = []
    #1.获取milvus的客户端
    milvus_client = get_milvus_client()
    #2.将item_names转换为向量库中的向量
    enbeddings = generate_embeddings(item_names)
    #3.在向量库中搜索,混合查询，稠密稀疏向量
    for index,item_name in enumerate(item_names):
        dense_vector = enbeddings["dense"][index]
        sparse_vector = enbeddings["sparse"][index]
        reqs = create_hybrid_search_requests(dense_vector,sparse_vector)
        response = hybrid_search(
            client=milvus_client,
            collection_name=milvus_config.item_name_collection,
            reqs = reqs,#请求体
            ranker_weights=(0.5,0.5),
            norm_score = True,#归一化打分
        )
    #4.提取查询结果，封装返回的数据格式
        matches = []
        if response and len(response) >0:
            for hit in response[0]:
                entity = hit.get("entity",{})
                hit_name = entity.get("item_name")
                score = hit.get("distance",0)
                if hit_name:
                    matches.append({"item_name":hit_name,"score":score})

        final_result.append({
            "extracted":item_name,
            "matches":matches
        })
    #5.封装返回数据
    return final_result


def step_5_confirm_and_optional_item_names(query_milvus_result):
    """
    对查找的结果进行打分处理，判断是否是确认集合或者是可选集合
    :param query_milvus_result:
    :return:result = {
        "confirmed_item_names":list(set(confirm_item_names)),
        "optional_item_names":list(set(optional_item_names))
    }
    """
    #准备两个列表，分别存储确认集合和可选集合
    confirm_item_names = []
    optional_item_names = []
    #循环处理元数据query_milvus_result
    for item_name_result in query_milvus_result:
        extracted_name = item_name_result.get("extracted")
        matches = item_name_result.get("matches",[])
        #进行分数排序（倒序）||高分》0.85，||0.6
        matches.sort(key = lambda x:x.get("score",0),reverse=True)
        #取出两个分数段》0.85||》0.6
        high_score_matches = [x for x in matches if x.get("score",0)>=0.85]
        middle_score_matches = [x for x in matches if x.get("score",0)>=0.6]

        #处理高分的列标，只有一个就获取第一个||有多个首选 item_name = extract 次选高分
        if len(high_score_matches) == 1:
            confirm_item_names.append(high_score_matches[0].get("item_name"))
            continue

        if len(high_score_matches) > 1:
            #首先寻找同名的 extract = item_name
            same_name_item = None
            for match in high_score_matches:
                if match.get("item_name") == extracted_name:
                    same_name_item = match
                    break
            if not same_name_item:
                #如果没有同名的 item_name，取第一个高分的
                same_name_item = high_score_matches[0]
                confirm_item_names.append(same_name_item.get("item_name"))
                continue


        #处理可选分数列标，给用户返回提示，可以选两个
        if len(middle_score_matches) > 0:
            for item in middle_score_matches[:2]:
                optional_item_names.append(item.get("item_name"))
            continue
        logger.info(f"没有高分和可选的item_name，忽略: {extracted_name}")
    result = {
        "confirmed_item_names":list(set(confirm_item_names)),
        "optional_item_names":list(set(optional_item_names))
    }
    logger.info(f"处理结果为: {result}")
    return result


def step_6_deal_list(state,item_results, history_chats,rewritten_query):
    """
    根据item_name确定的集合进行用户反馈结果的处理
    :param item_results:
    :param history_chats:
    :return:
    """
    #1.获取确认集合和可选集合
    confirmed_item_names = item_results.get("confirmed_item_names",[])
    optional_item_names = item_results.get("optional_item_names",[])
    #2.确认集合有数据直接处理
    if len(confirmed_item_names) > 0:
        state["item_names"] = confirmed_item_names
        state["rewritten_query"] = rewritten_query
        state["history"] = history_chats
        if "answer" in state:
            del state["answer"]
        logger.info(f"确认集合有数据，直接处理: {confirmed_item_names}")
        return state
    #3.确认集合没有数据，有可选集合，提示用户选择
    if len(optional_item_names) > 0:
        option_names = ",".join(optional_item_names)
        answer = f"您是向咨询一下哪个商品{option_names}？，请下次提问明确商品名称。"
        state["answer"] = answer
        logger.info(f"确认集合没有数据，有可选集合:{optional_item_names}")
        return state
    #4.确认集合没有数据，没有可选集合，提示用户重新输入
    answer = "没有匹配的选项，请重新输入。"
    state["answer"] = answer
    logger.info(f"确认集合没有数据，没有可选集合。")
    return state


def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    输入：state['original_query']
    输出：更新 state['item_names']
    """
    #1.提取用户问题item_names 在向量库中搜索 打分 取top_k个
    #2.利用模型重写用户的问题
    #核心参数：state['original_query'] ||session_id
    #返回值：item_names，rewrite_query,history
    #1.获取历史聊天记录
    #2.保存当次聊天记录
    #3.获取item_names，重写提问内容
    #4.在向量库中查找
    #5.对查找的结果进行打分处理，判断是否是确认集合或者是可选集合
    #6.处理确认可选集合，继续下个节点执行||有可选或者没有选择，为最后一个节点answer赋值
    #7.补充state状态
    print(f"---node_item_name_confirm---开始处理")
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    # 1.获取历史聊天记录
    history_chats = get_recent_messages(session_id=state["session_id"],limit=10)
    """
    为什么要重写
    1.消除代指歧义，比如用户问题中的他/她/它/她等
    2.结合历史消息，补全上下文，有实际的历史记录支撑
    3.去掉口语冗余
    4.润色问题增加召回率
    """
    # 3.获取item_names，重写提问内容
    item_name_and_rewrite_query = step_3_llm_item_name_and_rewrite_query(state["original_query"],history_chats)
    item_names = item_name_and_rewrite_query["item_names"]
    rewritten_query = item_name_and_rewrite_query.get("rewritten_query", "")
    # 4.在向量库中查找
    if len(item_names) > 0:
        query_milvus_result = step_4_query_milvus_item_names(item_names)
    # 5.对查找的结果进行打分处理，判断是否是确认集合或者是可选集合
    item_results = step_5_confirm_and_optional_item_names(query_milvus_result)
    #6根据item_name确定的集合进行用户反馈结果的处理
    state = step_6_deal_list(state,item_results,history_chats,rewritten_query)
    # # 记录本此的聊天记录
    # if state.get("answer"):
    #     #有answer就结束
    #     save_chat_message(
    #         session_id=state["session_id"],
    #         role="assistant",
    #         text=state["answer"],
    #         rewritten_query=rewritten_query,
    #         item_names=[],
    #     )

        # 2.保存档次聊天记录
    save_chat_message(
        session_id=state["session_id"],
        role="user",
        text=state["original_query"],
        rewritten_query=state.get("rewritten_query", ""),
        item_names=state.get("item_names", []),
        image_urls=[],
    )
    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    print(f"---node_item_name_confirm---处理结束")

    return state




if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "华为显示器好用吗",
        "is_stream": False
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        # 运行节点
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> 测试完成！最终状态:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False,default=str))

        # 简单验证
        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认商品名: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到商品名 (可能是向量库无匹配或LLM未提取)")

    except Exception as e:
        print(f"\n[FAIL] 测试运行出错: {e}")