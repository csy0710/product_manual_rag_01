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
    lines = md_content.splitlines('\n')
    is_code_block=False
    current_lines = []#当前标题行
    title_count=0#标题数量
    sections=[]#切分后的部分
    current_title=""#当前标题
    # 循环每行的列标
    for line in lines:
        strip_line = line.strip()

        #判断是不是代码块
        if strip_line.startwith('```') or strip_line.startwith('~~~'):
            # 如果是代码块，将代码块状态切换为True
            is_code_block=not is_code_block
            # 将当前行添加到当前标题行中
            current_lines = current_lines.append(line)
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
        separators=['\n\n','\n','。','!''，','，','']
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
    short_section = merge_short_section(final_sections,min_length)
    #补全属性和参数


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """
    logger.info(f">>> [Stub] 执行节点: {sys._getframe().f_code.co_name}")

    func_name = sys._getframe().f_code.co_name
    add_running_task(state["task_id"], func_name)

    try:
        #1.校验数据
        md_content,file_title = step_1_get_content(state)
        #2.粗粒度切割，语义完善-》使用标题切割[{content:标题内容,title:标题,file_title:文件名}]
        # 返回值：切分后的部分，标题数量，切分的数量
        sections,title_count,lines_count = c(md_content,file_title)
        # 3.特殊场景：无标题，给每个段落添加一个默认标题
        if title_count==0:
            sections = [{"title": "default_title", "content": md_content, "file_title": file_title}]
        # 4.细粒度切割，大小和重叠合适
        sections = step_3_split_by_content(sections,DEFAULT_MAX_CONTENT_LENGTH,MIN_CONTENT_LENGTH)

    except Exception as e:

        logger.error(f"【{func_name}】节点执行异常: {e}")


    add_done_task(state["task_id"], func_name)
    logger.info(f"【{func_name}】节点执行完成")


    return state