from typing import TypedDict
import copy
from app.core.logger import logger
class ImportGraphState(TypedDict):
    """
    图的状态定义，包含所有节点产生和消费的数据字段
    """
    task_id: str  #任务id


#路径相关
    local_file_path: str  #导入文件地址
    local_dir: str  #输出文件地址
    pdf_path: str  #pdf文件地址
    md_path: str  #markdown文件地址
    file_title: str  #文件名
    # split_path: str  #文档切割输出文件地址
    # embeding_path: str  #混合向量化输出文件地址
# 流程相关
    is_md_read_enabled:bool
    is_pdf_read_enabled:bool

# 内容数据
    md_content: str  #markdown文件的内容
    chunks: list  #每个文档的内容
    item_name: str  #识别到的主体名称

# # 切块相关
#     is_normal_split_enabled: bool
#     is_silicon_flow_api_enabled: bool
#     is_advanced_split_enabled: bool
#     is_vllm_enabled: bool

# 数据库相关
    embedding_content: list  #每个文档的向量化内容


# 建议定一个初始化对象，方便后续使用
# 定义图状态的默认初始值
graph_default_state: ImportGraphState = {
    "task_id":"",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "is_normal_split_enabled": True,
    "is_silicon_flow_api_enabled": True,
    "is_advanced_split_enabled": False,
    "is_vllm_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "split_path": "",
    "embeddings_path": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": []
}



def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖

    Args:
        **overrides: 要覆盖的字段（关键字参数解包）

    Returns:
        新的状态实例

    Examples:
        state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """

    # 默认状态
    state = copy.deepcopy(graph_default_state)
    # 用 overrides 覆盖默认值
    state.update(overrides)
    # 返回创建好的状态字典实例
    return state

def get_default_state() -> ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染
    """
    return copy.deepcopy(graph_default_state)


if __name__ == "__main__":
    """
    测试
    """
    # 创建默认状态
    state = create_default_state(local_file_path="万用表RS-12的使用.pdf")
    logger.info(state)

