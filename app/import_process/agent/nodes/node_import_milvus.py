import inspect
import os
import sys
from typing import List, Dict, Any
# 导入Milvus相关依赖
from pymilvus import DataType
# 导入自定义模块
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger
from app.conf.milvus_config import milvus_config

# 从配置文件读取切片集合名称，与配置解耦，便于环境切换
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection


def step_2_prepare_collection(state):
    """
    准备集合collection
    :param state: 包含输入数据的ImportGraphState对象
    :return: 连接Milvus的客户端对象
    """
    milvus_client = get_milvus_client()
    # 检查要创建的集合是否存在
    if not milvus_client.has_collection(collection_name=milvus_config.chunks_collection):
        # 创建集合，设置主键自增长，开启动态字段支持
        schema = milvus_client.create_schema(
            auto_id=False,  # 主键自增长
            enable_dynamic_field=True  # 开启动态字段支持
        )

        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65534)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65534)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65534)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65534)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65534)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 添加索引
        index_params = milvus_client.prepare_index_params()

        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="HNSW",
            metric_type="COSINE",
            params={
                "M": 32,
                "efConstruction": 300,
            }
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"}
        )
        milvus_client.create_collection(
            collection_name=milvus_config.chunks_collection,
            schema=schema,
            index_params=index_params
        )
    return milvus_client


def step_3_delete_old_data(milvus_client,item_name):
    """
    删除旧数据
    :param item_name: 包含输入数据的ImportGraphState对象
    :param milvus_client: 连接Milvus的客户端对象
    :return: None
    """
    # 删除旧数据
    milvus_client.delete(
        collection_name=CHUNKS_COLLECTION_NAME,
        filter = f"item_name=='{item_name}'"
    )
    # 提交删除
    milvus_client.load_collection(collection_name=CHUNKS_COLLECTION_NAME)


def step_4_insert_collection(chunks,milvus_client):
    """
    批量插入新的向量数据
    :param chunks: 包含输入数据的ImportGraphState对象
    :return: None
    """
    insert_result = milvus_client.insert(
        collection_name=CHUNKS_COLLECTION_NAME,
        data=chunks
    )
    # 提交插入
    insert_count = insert_result.get("insert_count",0)
    logger.info(f"成功插入 {insert_count} 条数据")

    #获取回显的ids
    ids = insert_result.get("ids",[])

    if ids and len(ids) == len(chunks):
        for index,chunk in enumerate(chunks):
            chunk["chunk_id"] = ids[index]


    return chunks

def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    未来要实现:
    1. 连接 Milvus。
    2. 根据 item_name 删除旧数据 (幂等性)。
    3. 批量插入新的向量数据。
    """

    func_name = inspect.currentframe().f_code.co_name
    logger.info(f">>> [Stub] 执行节点: {func_name}")
    add_running_task(state["task_id"], func_name)

    try:
        #校验数据
        chunks = state.get("chunks")
        if not chunks:
            logger.error("输入数据为空，无法导入向量库")
            raise ValueError("输入数据为空，无法导入向量库")
        #创建集合collection
        milvus_client = step_2_prepare_collection(state)
        #删除旧数据
        step_3_delete_old_data(milvus_client,chunks[0]["item_name"])
        #批量插入新的向量数据
        with_id_chunks = step_4_insert_collection(chunks,milvus_client)

        state['chunks'] = with_id_chunks
    except Exception as e:

        logger.error(f"【{func_name}】节点执行异常: {e}")
        raise
    finally:
        add_done_task(state["task_id"], func_name)
        logger.info(f"【{func_name}】节点执行完成,现在的state: {state}")
    return state






if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            },
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题2",
                "item_name": "测试项目_Milvus2",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf2",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"❌ 测试失败: {e}")