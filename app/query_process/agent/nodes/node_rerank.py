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
    scores = rerank.compute_score(question_pairs,normalization = True)
    #将原有数据添加对应的分
    doc_list_with_score = []
    #分别取分数和文档进行合并

    for score,item in zip(scores,doc_list):
        item["score"] = score
        doc_list_with_score.append(item)

    doc_list_with_score.sort(key=lambda x: x["score"], reverse=True)

    logger.info(f"已经完成了rerank排序，排序结果为: {doc_list_with_score}")
    return doc_list_with_score


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

    #4.返回state结果

    # ...
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))