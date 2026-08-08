import copy


from typing import List, TypedDict


class QueryGraphState(TypedDict):
    """
    QueryGraphState 定义了整个查询流程中流转的数据结构。
    """
    session_id: str  # 会话唯一标识
    original_query: str  # 用户原始问题

    # 检索过程中的中间数据
    embedding_chunks: list  # 普通向量检索回来的切片
    hyde_embedding_chunks: list  # HyDE 检索回来的切片
    kg_chunks: list  # 图谱检索回来的切片
    web_search_docs: list  # 网络搜索回来的文档

    # 排序过程中的数据
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档

    # 生成过程中的数据
    prompt: str  # 组装好的 Prompt
    answer: str  # 最终生成的答案

    # 辅助信息
    item_names: List[str]  # 提取出的商品名称
    rewritten_query: str  # 改写后的问题
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出标记

#默认状态，全部为空
query_default_state : QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "kg_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
}

#创建一个默认状态的QueryGraphState实例

def create_default_query_graph_state(**overrides) -> QueryGraphState:
    state = copy.deepcopy(query_default_state)
    state.update(overrides)
    return state

#获取干净状态
def get_query_default_state() -> QueryGraphState:
    return copy.deepcopy(query_default_state)



#状态复制函数

def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state










