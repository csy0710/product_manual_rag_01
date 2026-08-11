# HyDE节点
import sys

from langchain_core.messages import HumanMessage

from app.utils.task_utils import add_running_task, add_done_task
from app.lm.lm_utils import *
from app.lm.embedding_utils import *
from app.clients.milvus_utils import *
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


def step_1_create_hyde_doc(rewritten_query):
    """
    生成假设性答案
    :param rewritten_query: 重写后的查询
    :return: 假设性答案
    """
    # 调用llm模型生成假设性答案
    llm = get_llm_client()
    #加载提示词
    hyde_prompt = load_prompt("hyde_prompt",rewritten_query = rewritten_query)
    messages = [
        HumanMessage(content = hyde_prompt)
    ]

    response = llm.invoke(messages)
    hyde_doc = response.content
    logger.info(f"问题：{rewritten_query} 生成的假设性答案：{hyde_doc}")
    return hyde_doc


def step_2_search_embedding(rewritten_query, hyde_doc, item_names):
    """
    对假设性答案进行向量检索
    :param rewritten_query: 重写后的查询
    :param hyde_doc: 假设性答案
    :param item_names: 要检索的文档名称列表
    :return:
    """
    #拼接重写问题+lm生成的假设性答案
    query_str = rewritten_query + hyde_doc
    embeddings= generate_embeddings([query_str])
    #生成查询AnnSearchRequest
    item_names_str = ','.join(f'"{item}"' for item in item_names)
    reqs = create_hybrid_search_requests(
        dense_vector=embeddings['dense'][0],
        sparse_vector=embeddings['sparse'][0],
        expr = f"item_name in [{item_names_str}]"
    )

    #进行混合向量检索
    milvus_client = get_milvus_client()
    resp = hybrid_search(
        client = milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs = reqs,
        ranker_weights = (0.9,0.1),
        output_fields=["item_name","content","title","parent_title","chunk_id"]
    )
    #处理返回的结果
    result = resp[0] if resp else []
    logger.info(f"检索到的结果：{result}")
    return result


def node_search_embedding_hyde(state):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    print("---HyDE 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 提取参数
    rewritten_query = state["rewritten_query"]
    item_names = state["item_names"]
    #调用llm模型生成假设性答案
    hyde_doc = step_1_create_hyde_doc(rewritten_query)
    #问题+答案，进行向量化检索
    resp = step_2_search_embedding(rewritten_query,hyde_doc,item_names)
    # ...
    #赋值和返回结果
    # state["hyde_embedding_chunks"] = resp
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    #
    print("---HyDE 处理结束---")
    return {"hyde_embedding_chunks": resp}


if __name__ == "__main__":
    # 本地测试代码
    print("\n" + "=" * 50)
    print(">>> 启动 node_search_embedding_hyde 本地测试")
    print("=" * 50)

    # 模拟输入状态
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "HAK 180 烫金机怎么操作？",
        "rewritten_query": "HAK 180 烫金机的具体操作步骤是什么？",
        "item_names": ["HAK 180 烫金机"],
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_search_embedding_hyde(mock_state)

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"HyDE Doc Generated: {bool(result.get('hyde_doc'))}")
        if result.get("hyde_doc"):
            print(f"Doc Preview: {result.get('hyde_doc')[:50]}...")

        chunks = result.get("hyde_embedding_chunks", [])
        print(f"Chunks Found: {len(chunks)} , chunks内容：{chunks}")
        if chunks:
            print(f"Top Chunk Score: {chunks[0].get('distance')}")
        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")