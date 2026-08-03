import os.path
import sys
from os.path import splitext
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.format_utils import format_state

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    未来要实现:
    1. 接收文件路径。
    2. 判断文件类型 (PDF/MD)。
    3. 设置 state 中的路由标记 (is_pdf_read_enabled / is_md_read_enabled)。
    """
    # 1.动态获取函数名称
    func_name = sys._getframe().f_code.co_name
    logger.info(f"节点名称: {func_name}")
    #2.开始记录节点状态
    add_running_task(state["task_id"],func_name)
    # 节点启动日志，打印当前工作流状态
    logger.info(f"【{func_name}】节点启动，\n当前工作流状态：{format_state(state)}")



    #3.核心参数提取与非空校验
    document_path = state["local_file_path"]

    if not document_path:
        logger.error(f"节点 {func_name} 缺少必要参数: local_file_path,文件路径为空")
        return state

    if document_path.endswith(".pdf"):
        logger.info(f"文件类型校验通过: {document_path} 为 PDF 文件，开启PDF解析流程")
        state["is_pdf_read_enabled"] = True
        # 将pdf文件地址记录到state中，方便后续节点使用
        state["pdf_path"] = document_path
    elif document_path.endswith(".md"):
        logger.info(f"文件类型校验通过: {document_path} 为 Markdown 文件，开启Markdown解析流程")
        state["is_md_read_enabled"] = True
        # 将markdown文件地址记录到state中，方便后续节点使用
        state["md_path"] = document_path
    else:
        logger.warning(f"文件类型校验未通过: {document_path} 不支持的格式，仅支持PDF和Markdown文件")



    #4.提取文件无后缀纯名称，作为全局业务标识
    file_name = os.path.basename(document_path)
    logger.info(f"文件无后缀纯名称: {file_name}")
    state["file_title"]= splitext(file_name)[0]
    logger.info(f"【{func_name}】文件业务标识提取完成：file_title = {state['file_title']}")


    # 5.结束，记录节点运行状态打印当前工作流状态
    func_name = sys._getframe().f_code.co_name
    add_done_task(state["task_id"],func_name)
    logger.info(f"【{func_name}】节点执行完成，\n更新后工作流状态：{format_state(state)}")
    return state
    # # 模拟简单的路由逻辑，防止报错 (仅 node_entry 需要)
    # if "local_file_path" in state:
    #     path = state["local_file_path"]
    #     if path.endswith(".pdf"):
    #         state["is_pdf_read_enabled"] = True
    #     elif path.endswith(".md"):
    #         state["is_md_read_enabled"] = True
    #
    # return state

# if __name__ == '__main__':
#
#         # 单元测试：覆盖不支持类型、MD、PDF三种场景
#         logger.info("===== 开始node_entry节点单元测试 =====")
#
#         # 测试1: 不支持的TXT文件
#         test_state1 = create_default_state(
#             task_id="test_task_001",
#             local_file_path="联想海豚用户手册.txt"
#         )
#         node_entry(test_state1)
#
#         # 测试2: MD文件
#         test_state2 = create_default_state(
#             task_id="test_task_002",
#             local_file_path="小米用户手册.md"
#         )
#         node_entry(test_state2)
#
#         # 测试3: PDF文件
#         test_state3 = create_default_state(
#             task_id="test_task_003",
#             local_file_path="万用表的使用.pdf"
#         )
#         node_entry(test_state3)
#
#         logger.info("===== 结束node_entry节点单元测试 =====")

