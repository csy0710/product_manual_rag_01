
import sys

from app.conf.milvus_config import milvus_config
from app.query_process.agent.state import QueryGraphState
import sys
import os
from app.utils.task_utils import add_running_task,add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests,hybrid_search,get_milvus_client
from app.core.logger import logger
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

def node_search_embedding(state):
    """
    节点: 搜索嵌入 (node_search_embedding)
    为什么叫这个名字: 核心任务是根据用户问题中的商品名称，从嵌入向量数据库中检索最相关的文档。
    未来要实现:
    1. 调用嵌入向量数据库 (如 milvus) 进行搜索。
    2. 将搜索结果保存到 {"embedding_chunks":[chunks]}。
    3.需要的参数：rewritten_query（用户重写问题中的商品名称）,item_names（用户确认的商品名称）
    """
    #先从state获取参数列表
    rewritten_query = state["rewritten_query"]
    item_names = state["item_names"]
    item_name_str = ','.join(f"'{item}'" for item in item_names)
    #将重写的问题生成对应的向量
    embedding = generate_embeddings([rewritten_query])
    #进行向量数据库的混合检索
    hybird_search_requests = create_hybrid_search_requests(
        dense_vector=embedding["dense"][0],
        sparse_vector=embedding["sparse"][0],
        expr=f"item_name in [{item_name_str}]"
    )
    #进行混合查询触发
    milvus_client = get_milvus_client()

    resp = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=hybird_search_requests,
        ranker_weights=(0.9,0.1),
        norm_score = True,
        limit=5,
        output_fields=["chunk_id","content","file_title","title","parent_title","item_name"]
       )
    """
    结果的格式：
    [
        [
            id,
            distance,
            entity:
            {
            "chunk_id",
            "content",
            "file_title",
            "title",
            "parent_title",
            "item_name"
            },
        
        ]
    ]
    """
    #处理查询结果赋值
    embedding_chunks = resp[0] if resp else []


    print(f">>> [Stub] 执行节点: {sys._getframe().f_code.co_name}")
    return {"embedding_chunks": embedding_chunks}





if __name__ == "__main__":
    # 模拟测试数据
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "HAK 180 烫金机使用说明",  # 模拟改写后的查询
        "item_names": ["HAK 180 烫金机"],  # 模拟已确认的商品名
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        # 执行节点函数
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果，结果为{chunks}")

        if chunks:
            print("\n>>> Top 1 结果详情:")
            top1 = chunks[0]
            # 打印关键字段（注意：entity字段可能包含具体业务数据）
            print(f"ID: {top1.get('id')}")
            print(f"Distance: {top1.get('distance')}")
            entity = top1.get('entity', {})
            print(f"Item Name: {entity.get('item_name')}")
            print(f"Content Preview: {entity.get('content', '')[:100]}...")
        else:
            print("\n>>> 警告：未检索到任何结果，请检查 Milvus 数据或 item_names 是否匹配")

    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)