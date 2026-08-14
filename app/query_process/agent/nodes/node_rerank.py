import sys
from app.utils.task_utils import *

from dotenv import load_dotenv
import sys
from app.lm.reranker_utils import get_reranker_model
from app.utils.task_utils import add_running_task
from app.core.logger import logger
load_dotenv()

# -----------------------------
# Rerank / TopK 全局常量（不从 state 读取）
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对）
RERANK_GAP_RATIO: float = 0.25
# 断崖阈值（绝对）
RERANK_GAP_ABS: float = 0.5


def step_1_merge_rrf_mcp(state):
    """
    进行rrf和mcp的数据整合
    :param state:
    :return:
    """
    #1.获取state中的数据
    rrf_chunks = state.get("rrf_chunks", [])
    web_search_docs = state.get("web_search_docs", [])
    #准备一个容器
    chunks_list = []
    # 循环进行数据添加
    for chunk in rrf_chunks:
        entity = chunk.get("entity")
        chunk_id = entity.get("chunk_id")
        content = entity.get("content")
        title = entity.get("title")
        chunks_list.append({
            "chunk_id": chunk_id,
            "text": content,
            "title": title,
            "source":"local",
            "url":"",
        })
    #mcp
    for doc in web_search_docs:
        text = doc.get("snippet")
        url = doc.get("url")
        title = doc.get("title")
        chunks_list.append({
            "chunk_id": "",
            "text": text,
            "title": title,
            "source":"web",
            "url":url,
        })
    logger.info(f"多路数据合并结果为: {chunks_list}")
    return chunks_list


def step_2_rerank_doc_list(doc_list, state):
    """
    对文档列表进行rerank排序
    :param doc_list:
    :param state:
    :return:
    """
    #1.获取原有问题
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    #2.获取对应的答案
    text_list = [doc["text"] for doc in doc_list]
    #3.拼接参数
    question_pairs = [[rewritten_query, text] for text in text_list]
    #4.调用rerank模型进行排序
    rerank = get_reranker_model()
    #返回值是一个分数列表
    scores = rerank.compute_score(question_pairs,normalize = True)
    #将原有数据添加对应的分
    doc_list_with_score = []
    #分别取分数和文档进行合并

    for score,item in zip(scores,doc_list):
        item["score"] = score
        doc_list_with_score.append(item)

    doc_list_with_score.sort(key=lambda x: x["score"], reverse=True)

    logger.info(f"已经完成了rerank排序，排序结果为: {doc_list_with_score}")
    return doc_list_with_score


def step_3_topk_and_gap(rerank_score_list):
    """
    对rerank排序后的文档列表进行断崖以及topk处理
    :param rerank_score_list:
    :return:
    """
    max_topk = RERANK_MAX_TOPK #最多取前N条
    min_topk = RERANK_MIN_TOPK #至少保留前N条
    gap_ratio = RERANK_GAP_RATIO#相对断崖阈值
    gap_abs = RERANK_GAP_ABS#绝对断崖阈值
    #top不应该大于列表长度
    topk = min(max_topk,len(rerank_score_list))
    #循环处理数据列表，进行双指针处理和比较
    if topk >= min_topk:
        for index in range(min_topk-1,topk-1):
            #双指针处理
            score_1 = rerank_score_list[index].get("score",0.0)
            score_2 = rerank_score_list[index+1].get("score",0.0)
            #计算分数的差值gap
            gap = score_1 - score_2

            rel = gap/(abs(score_1)+1e-6)
            if gap>=gap_abs or rel>=gap_ratio:
                logger.info(f"断崖阈值（绝对）>0.5 取前{index+1}条")
                topk = index + 1
                break

    topk_doc_list = rerank_score_list[:topk]
    logger.info(f"最终取前{topk}条,截取的内容为: {topk_doc_list}")

    return topk_doc_list




def node_rerank(state):
    """
    节点功能：对网络搜索的答案和rrf融合后搜索的答案进行rerank非同源排序（使用模型的能力进行打分和分析）
    rrf+mcp -> rerank ->chunk打分 ->算法 ->top_k
    算法理解：算法（最多十条最少一条 相对0.25 绝对0.5）
    0.93，0.91，0.90,0.39..... ->断崖阈值（绝对）>0.5 取前三条
    0.4,0.3,0.2 -> (0.4-0.3)/0.4=相对断崖阈值<0.25 取0.4,0.3   ->（0.3-0.2）/0.3=相对断崖阈值>0.25 不取0.2

    """
    print("---Rerank处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    #1.非同源路的结果合并
    doc_list = step_1_merge_rrf_mcp(state)
    #2.启用rerank进行精排
    rerank_score_list = step_2_rerank_doc_list(doc_list,state)
    #3.启动算法进行断崖以及topk处理
    final_doc_list = step_3_topk_and_gap(rerank_score_list)
    #4.返回state结果
    state["reranked_docs"] = final_doc_list

    # ...
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rerank 本地测试")
    print("=" * 50)

    # 1. 模拟数据
    # 1.1 RRF 本地文档数据
    mock_rrf_chunks = [
        {"entity":{"chunk_id": "local_1", "content": "RRF是一种倒数排名融合算法", "title": "算法介绍", "score": 0.9}},
        {"entity":{"chunk_id": "local_2", "content": "BGE是一个强大的重排序模型", "title": "模型介绍", "score": 0.8}},
        {"entity":{"chunk_id": "local_3", "content": "无关的测试文档内容", "title": "测试文档", "score": 0.1}}  # 预期低分
    ]

    # 1.2 MCP 联网搜索数据
    mock_web_docs = [
        {"title": "Rerank技术详解", "url": "http://web.com/1", "snippet": "Rerank即重排序，常用于RAG系统的第二阶段"},
        {"title": "无关网页", "url": "http://web.com/2", "snippet": "今天天气不错，适合出去游玩"}  # 预期低分
    ]

    mock_state = {
        "session_id": "test_rerank_session",
        "rewritten_query": "什么是RRF和Rerank？",  # 查询意图：想了解这两个算法
        "rrf_chunks": mock_rrf_chunks,
        "web_search_docs": mock_web_docs,
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_rerank(mock_state)
        reranked = result.get("reranked_docs", [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入文档总数: {len(mock_rrf_chunks) + len(mock_web_docs)}")
        print(f"输出文档总数: {len(reranked)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(reranked, 1):
            print(f"Rank {i}: Source={doc.get('source')}, Score={doc.get('score'):.4f}, Text={doc.get('text')[:20]}...")

        # 验证逻辑：
        # 预期 "local_1", "local_2", "Rerank技术详解" 分数较高
        # 预期 "local_3", "无关网页" 分数较低，可能被截断或排在最后

        top1_score = reranked[0].get("score")
        if top1_score > 0:
            print("\n[PASS] Rerank 打分正常")
        else:
            print("\n[FAIL] Rerank 打分异常 (均为0或负数)")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")