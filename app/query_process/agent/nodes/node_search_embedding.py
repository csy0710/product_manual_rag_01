
import sys

from app.query_process.agent.state import QueryGraphState


def node_search_embedding(state):
    """
    节点: 搜索嵌入 (node_search_embedding)
    为什么叫这个名字: 核心任务是根据用户问题中的商品名称，从嵌入向量数据库中检索最相关的文档。
    未来要实现:
    1. 调用嵌入向量数据库 (如 ChromaDB) 进行搜索。
    2. 将搜索结果保存到 state["reranked_docs"]。
    """

    print(f">>> [Stub] 执行节点: {sys._getframe().f_code.co_name}")
    return {"embedding_chunks": []}
