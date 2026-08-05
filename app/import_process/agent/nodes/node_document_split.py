import re
import json
import os
import sys
# 统一类型注解，避免混用any/Any
from typing import List, Dict, Any, Tuple
# LangChain文本分割器（标注核心用途，便于理解）
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 项目内部工具/状态/日志导入（保持原有路径）
from app.utils.task_utils import add_running_task, add_done_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger  # 项目统一日志工具，核心替换print

# --- 配置参数 (Configuration) ---
# 单个Chunk最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
DEFAULT_MAX_CONTENT_LENGTH = 2000
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500

"""
完成md内容的切分，将长文档切分成小的 Chunks (切片) 以便检索。
    chunks-》存储块的集合  chunks-》备份到本地-》chunks.json
    1.参数校验
    2.粗粒度切割，语义完善-》使用标题切割
    3.特殊场景：无标题，给每个段落添加一个默认标题
    4.细粒度切割，大小和重叠合适
    5.数据的备份和chunks属性的修改
"""


def step_1_get_content(state):
    md_content = state["md_content"]
    if not md_content:
        logger.error(f"[step_1_get_content]没有有效的md_content内容")
        raise ValueError("md_content不能为空")
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    file_title = state.get("file_title", "default_title")
    return md_content,file_title


def step_2_split_by_title(md_content, file_title):
    """语义切割，基于标题层级进行递归切分
    :param md_content: 输入的 Markdown 内容。
    :param file_title: 文件标题，用于生成默认标题。
    :return: 切分后的标题列表 (包含默认标题)content，标题title，文件名file_title。
    """
    #定义一个正则规则
    title_pattern = r'^\s*#{1,6}\s+.+'
    lines = md_content.split('\n')
    is_code_block=False
    current_lines = []#当前标题行
    title_count=0#标题数量
    sections=[]#切分后的部分
    current_title=""#当前标题
    # 循环每行的列标
    for line in lines:
        strip_line = line.strip()

        #判断是不是代码块
        if strip_line.startswith('```') or strip_line.startswith('~~~'):
            # 如果是代码块，将代码块状态切换为True
            is_code_block=not is_code_block
            # 将当前行添加到当前标题行中
            current_lines.append(line)
            continue
        # 判断是不是标题 需要代码块状态为False 且符合标题规则
        is_title = (not is_code_block) and re.match(title_pattern,strip_line)
        #如果是标题
        if is_title:
            # 如果当前标题不为空，说明是第一个标题之后的标题，将当前标题添加到切分后的部分
            if current_title:
                sections.append(
                    {"title": current_title,
                     "content": "\n".join(current_lines),
                     "file_title": file_title
                     }
                )

            #如果是标题且当前标题为空证明是第一个标题，将当前标题添加到切分后的部分
            current_title = strip_line
            current_lines = [current_title]
            title_count+=1
        else:
            # 如果不是标题，将当前行添加到当前标题行中
            current_lines.append(line)

    # 如果当前标题不为空，说明是最后一个标题，将当前标题添加到切分后的部分
    if current_title:
        sections.append(
            {"title": current_title,
             "content": "\n".join(current_lines),
             "file_title": file_title
             }
        )
    logger.info(f"已经完成chunks的语义粗切，识别chunk数量：{title_count}")

    return sections,title_count,len(lines)

def split_long_section(section, max_length):
    """
    切分段落，超过最大长度的段落需要二次切分
    返回切割后的[{},{},{}]
    :param section: 段落内容
    :param max_length: 最大长度
    :return: 切分后的段落内容
    """
    #content获取
    content = section.get("content")
    #判断content是否超过最大长度
    if len(content) <= max_length:
        logger.info(f"段落{content}长度小于等于最大长度{max_length}，无需切分")
        return [section]
    #超过最大长度，需要二次切分
    sub_sections = []
    spliter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=100,
        separators=['\n\n','\n','。','!','，','']
    )
    for index,trunk in enumerate(spliter.split_text(content),start=1):
        text = trunk.strip()#切片的内容
        title = f"{section.get('title')}_{index}"
        file_title = section.get("file_title")
        part = index
        parent_title = section.get("title")
        sub_sections.append(
            {
                "title":title,
                "content":text,
                "file_title":file_title,
                "parent_title":parent_title,
                "part": part,
             }
        )
    # 返回切割后的内容
    return sub_sections


def merge_short_section(final_sections, min_length):
    """
    上一次切的太碎，需要合并到上一个段落
    1.content长度小于min_length
    2.同一个parent_title才能合并
    :param final_sections:
    :param min_length:
    :return:
    """
    merge_sections=[]
    pre_section=None
    #循环处理问题
    for section in final_sections:
        #第一次进来为上一次赋值
        if pre_section is None:
            pre_section = section
            continue
        is_current_short = len(pre_section.get("content")) < min_length#判断上一次是不是短块需要合并
        is_same_parent_title = pre_section.get("parent_title") == section.get("parent_title")#判断是否是同一个parent_title


        if is_current_short and is_same_parent_title:
            # 上一次是短块，且是同一个parent_title，需要合并到上一个段落
            current_content = section.get("content")
            pre_section["content"] +="\n\n" + current_content#合并内容
            pre_section["part"]  = section.get("part")#更新这个段落的part为当前段落的part

        else:
            # 上一次不是短块，或者不是同一个parent_title，需要合并到上一个段落
            merge_sections.append(pre_section)
            pre_section = section#本次段落在下个循环开始作为上一次段落
    #循环外处理
    if pre_section is not None:
        merge_sections.append(pre_section)


    return merge_sections

def step_3_split_by_content(sections,max_length,min_length):
    """做内容的精细切割
    1.超过了最大长度，需要二次切分
    2.如果段落小于最小长度，需要合并到上一个段落
    :param sections:
    :param max_length:
    :param min_length:
    :return:sections
    """
    final_sections = [] #最终的切分后的部分
    #超过的先切碎
    for section in sections:
        sub_section = split_long_section(section,max_length)
        final_sections.extend(sub_section)#平铺切割后的的内容到最终的切分后的部分 如果使用append会成为 [[],[],[]]
    #小于的再合并
    final_sections = merge_short_section(final_sections,min_length)
    #补全属性和参数
    for section in final_sections:
        section["part"] = section.get('part') or 1
        section["parent_title"] = section.get('parent_title') or section.get('title')

    return final_sections


def step_4_backup_chunks(state, sections):
    """
    备份切分后的段落到数据库
    :param state:取local_dir
    :param sections:
    :return:
    """
    local_dir = state.get("local_dir")
    backup_file_path = os.path.join(local_dir,"chunks.json")
    with open(backup_file_path, "w",encoding="utf-8") as f:
        json.dump(
            sections,#将什么数据写入文件
            f, #写入的位置
            ensure_ascii=False,#中文直接原文存储
            indent=4#缩进4个空格
        )
    logger.info(f"已经将备份切分后的段落到文件{backup_file_path}")

def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    思路：
    1. 校验数据
    2. 粗粒度切割，语义完善-》使用标题切割[{content:标题内容,title:标题,file_title:文件名}]
    3. 特殊场景：无标题，给每个段落添加一个默认标题
    4. 细粒度切割，大小和重叠合适
    5. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """
    logger.info(f">>> [Stub] 执行节点: {sys._getframe().f_code.co_name}")

    func_name = sys._getframe().f_code.co_name
    add_running_task(state["task_id"], func_name)

    try:
        #1.校验数据
        md_content,file_title = step_1_get_content(state)
        #2.粗粒度切割，语义完善-》使用标题切割[{content:标题内容,title:标题,file_title:文件名}]
        # 返回值：切分后的部分，标题数量，切分的数量
        sections,title_count,lines_count = step_2_split_by_title(md_content,file_title)
        # 3.特殊场景：无标题，给每个段落添加一个默认标题
        if title_count==0:
            sections = [{"title": "default_title", "content": md_content, "file_title": file_title}]
        # 4.细粒度切割，大小和重叠合适
        sections = step_3_split_by_content(sections,DEFAULT_MAX_CONTENT_LENGTH,MIN_CONTENT_LENGTH)
        # 5.生成包含 Metadata (标题路径) 的 Chunk 列表。
        state["chunks"] = sections
        step_4_backup_chunks(state,sections)
    except Exception as e:

        logger.error(f"【{func_name}】节点执行异常: {e}")


    add_done_task(state["task_id"], func_name)
    logger.info(f"【{func_name}】节点执行完成")


    return state





if __name__ == '__main__':
    """
    单元测试：联合node_md_img（图片处理节点）进行集成测试
    测试条件：1.已配置.env（MinIO/大模型环境） 2.存在测试MD文件 3.能导入node_md_img
    测试流程：先运行图片处理→再运行文档切分，验证端到端流程
    """

    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # 构造测试状态对象，模拟流程入参
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir":os.path.join(PROJECT_ROOT, "output"),
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
        logger.info("\n=== 开始执行文档切分节点集成测试 ===")

        logger.info(">> 开始运行当前节点：node_document_split（文档切分）")
        final_state = node_document_split(result_state)
        final_chunks = final_state.get("chunks", [])
        logger.info(f"✅ 测试成功：最终生成{len(final_chunks)}个有效Chunk{final_chunks}")