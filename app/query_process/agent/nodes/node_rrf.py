import sys
from typing import List, Dict, Any
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger

def step_3_reciporcal_rank_fusion(source_with_weight,top_k=5):
    """
    3. 反向排名融合
    :param source_with_weight: 来源与权重的列表，每个元素为(source_name,weight)
    :param source_with_weight: 源数据的列表，每个元素为{id,distance,entity{chunk_id,content,title...}}
    :return: 融合后的数据列表
    """
    #1.准备两个容器 记录历史得分
    score_dict = {}
    chunk_dict = {}
    # source_with_weight[
    #     ("embedding",1),
    #     ("hyde_embedding",1)
    # ]

    #循环处理每个集合中的数据
    for source,weight in source_with_weight:
        #source = [{id,distance,entity{chunk_id,content,title...}},{},{}]
        for rank,chunk in enumerate(source, start=1):
            # 计算当前chunk的得分
            chunk_id = chunk.get("id") or chunk.get("entity").get("chunk_id")
            score_dict[chunk_id]=score_dict.get(chunk_id,0.0) + (1.0/(60+rank)*weight)
            # chunk_dict[chunk_id] = chunk
            chunk_dict.setdefault(chunk_id,chunk)

    #分和chunk的融合加排序
    merged=[]
    for chunk_id,score in score_dict.items():
        chunk = chunk_dict.get(chunk_id)
        merged.append((chunk,score))
    merged.sort(key=lambda x:x[1],reverse=True)
    merged = merged[:top_k]
    rank_chunks = [chunk for chunk,score in merged]
    logger.info(f"完成了rrf排名融合，融合后的数据为{rank_chunks}")
    return rank_chunks
def node_rrf(state):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web、KG）进行加权融合排序。
    """
    print("---RRF---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    # 1.获取同源路的数据
    # [{id,distance,entity{chunk_id,content,title...}},{},{}]
    embedding_chunks = state.get("embedding_chunks")
    hyde_embedding_chunks = state.get("hyde_embedding_chunks")
    # 2.数据整合
    source_with_weight = [
        (embedding_chunks, 1),
        (hyde_embedding_chunks, 1)
    ]
    rrf_response = step_3_reciporcal_rank_fusion(source_with_weight)
    state["rrf_chunks"] = rrf_response
    # ...
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rrf 本地测试")
    print("=" * 50)

    # 1. 构造假数据 (模拟真实数据库字段)
    # 模拟 Embedding 检索结果
    mock_embedding_chunks = [
        {
            "id": "doc_1",
            "pk": "pk_1",
            "file_title": "操作手册_v1.pdf",
            "item_name": "HAK 180 烫金机",
            "content": "内容1：打开电源开关...",
            "score": 0.9
        },
        {
            "id": "doc_2",
            "pk": "pk_2",
            "file_title": "维修指南.pdf",
            "item_name": "HAK 180 烫金机",
            "content": "内容2：遇到故障请联系...",
            "score": 0.8
        },
        {
            "id": "doc_3",
            "pk": "pk_3",
            "file_title": "参数表.xlsx",
            "item_name": "HAK 180 烫金机",
            "content": "内容3：电压220V...",
            "score": 0.7
        }
    ]

    # 模拟 HyDE 检索结果 (包含 3 个文档，顺序不同，且有新文档 doc_4)
    mock_hyde_chunks = [
        {
            "id": "doc_3",
            "pk": "pk_3",
            "file_title": "参数表.xlsx",
            "item_name": "HAK 180 烫金机",
            "content": "内容3：电压220V...",
            "score": 0.85
        },
        {
            "id": "doc_1",
            "pk": "pk_1",
            "file_title": "操作手册_v1.pdf",
            "item_name": "HAK 180 烫金机",
            "content": "内容1：打开电源开关...",
            "score": 0.82
        },
        {
            "id": "doc_4",
            "pk": "pk_4",
            "file_title": "安全须知.docx",
            "item_name": "HAK 180 烫金机",
            "content": "内容4：操作时请佩戴手套...",
            "score": 0.75
        }
    ]

    # 模拟输入状态
    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "embedding_chunks": mock_embedding_chunks,
        "hyde_embedding_chunks": mock_hyde_chunks
    }

    try:
        # 运行节点
        result = node_rrf(mock_state)

        # 验证结果
        rrf_chunks = result.get("rrf_chunks", [])
        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入数量: Embedding={len(mock_embedding_chunks)}, HyDE={len(mock_hyde_chunks)}")
        print(f"输出数量: {len(rrf_chunks)}")
        print("-" * 30)

        # 打印详细排名
        print("最终排名:")
        for i, doc in enumerate(rrf_chunks, 1):
            # 注意：返回结果中可能没有 chunk_id 字段，而是 id
            doc_id = doc.get('chunk_id') or doc.get('id')
            print(f"Rank {i}: ID={doc_id}, Title={doc.get('file_title')}, Content={doc.get('content')[:20]}...")

        # 验证预期逻辑：
        ids = [d.get("id") or d.get("chunk_id") for d in rrf_chunks]

        if "doc_1" in ids and "doc_3" in ids:
            print("\n[PASS] 交叉文档 (doc_1, doc_3) 成功融合保留")
        else:
            print("\n[FAIL] 交叉文档丢失")

        if len(ids) == 4:
            print("[PASS] 并集数量正确 (3+3-2重叠=4)")
        else:
            print(f"[FAIL] 并集数量错误: 期望4, 实际{len(ids)}")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")