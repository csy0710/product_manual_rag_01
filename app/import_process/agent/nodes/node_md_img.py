import os
import sys
from pathlib import Path
from typing import Tuple, List

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
import os
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

# MinIO相关依赖
from minio import Minio
from minio.deleteobjects import DeleteObject

# 【核心改造1：移除原生OpenAI，导入LangChain工具类和多模态消息模块】
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task
# LLM客户端工具类（核心复用，替换原生OpenAI调用）
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖（消息构造+异常捕获）
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
# 项目配置
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目日志工具（统一使用）
from app.core.logger import logger
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt

"""
     节点: 图片处理 (node_md_img)
     为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
     未来要实现:
     1. 扫描 Markdown 中的图片链接。
     2. 将图片上传到 MinIO 对象存储。
     3. (可选) 调用多模态模型生成图片描述。
     4. 替换 Markdown 中的图片链接为 MinIO URL。



     1.校验并获取本次操作的数据
     参数：state   ->md_path,md_content
     返回值：校验后的md_content，md路径对象，图片路径对象
     2.识别md中使用过的图片
     参数：md_content,图片路径地址
     返回值：[（图片名，图片地址，（上文，下文））]
     3.进行图片内容的处总结和处理
     参数：[（图片名，图片地址，（上文，下文））]，md的文件名称
     返回值：[（图片名，图片描述）]
     4.上传图片到minIO，更新md的内容
     参数：minio_client,{图片名：图片描述}，[（图片名，图片地址，（上文，下文））]，md_content,md文件名称
     返回值：更新后的md_content
     5.进行数据的处理和备份
     参数：new_md_content,md_path,
     返回值：new_md_path
"""


# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
def is_supported_image(filename: str) -> bool:
    """ 判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
            :param filename: 文件名（含后缀）
            :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:

    """提取内容"""
    # 字符串是否为空
    md_file_path = state["md_path"]
    if not md_file_path:
        raise ValueError("md_path不能为空")
    md_path_obj = Path(md_file_path)
    # 文件是否存在
    if not md_path_obj.exists():
        raise FileNotFoundError(f"md_path {md_file_path} 不存在")
    # 读取文件内容
    if not state["md_content"]:
        #没有再读取，有证明是pdf节点解析过来的。
        with open(md_file_path, "r", encoding="utf-8") as f:
            md_content = f.read()
        state["md_content"] = md_content


    # 图片路径是否为空
    # 自己传入的md文件中的图片文件夹也要叫images，这里没有特殊处理
    images_dir_obj =md_path_obj / "images"

    return md_content,md_path_obj,images_dir_obj


def find_image_in_md_content(md_content, image_file,context_length:int=100):
    """在md中查找图片上下文"""
    # 定义正则表达式
    pattern = re.compile(r"!\[.*?\(.*?"+image_file+".*?\)")

    results = [] #存储图片多次使用，上下文不同！本次暴力处理，获取第一个
    # 查询符合位置
    for item in pattern.finditer(md_content):
        start,end = item.span()#获取的是匹配的图片链接的起始位置和结束位置
        #截取上文
        pre_text = md_content[max(start-context_length,0):start]
        #截取下文
        post_text = md_content[end:min(end+context_length,len(md_content))]
        results.append((pre_text,post_text))
    # 截取位置前后内容
    if results:
        logger.info(f"【find_image_in_md_content】节点执行完成，图片 {image_file} 在md中使用，截取第一个上下文为：{results[0]}")
        return results[0]


def step_2_scan_images(md_content:str,images_dir_obj:Path) -> List[Tuple[str,str,]]:
    """扫描md中使用过的图片,并且截取图片上下文"""
    targets=[]
    #循环读取images中的所有图片，校验在md中是否使用，使用了就截取上下文
    for image_file in os.listdir(images_dir_obj):
        if not is_supported_image(image_file):
            logger.info(f"【step_2_scan_images】节点执行完成，图片 {image_file} 不是支持的图片格式")
            continue

        content_data = find_image_in_md_content(md_content,image_file)
        if not content_data:
            logger.warning(f"【step_2_scan_images】节点执行完成，图片 {image_file} 未在md中使用,上下文为空")
            continue



        targets.append((image_file,(images_dir_obj/image_file),content_data))


    return targets



def node_md_img(state: ImportGraphState) -> ImportGraphState:

    logger.info(f">>> [Stub] 执行节点: {sys._getframe().f_code.co_name}")
    func_name = sys._getframe().f_code.co_name
    add_running_task(state["task_id"], func_name)
    #  1.校验并获取本次操作的数据
    #     参数：state   ->md_path,md_content
    #     返回值：校验后的md_content，md路径对象，图片路径对象
    md_content,md_path_obj,images_dir_obj = step_1_get_content(state)
    # 如果没有图片文件夹，直接返回
    if not images_dir_obj.exists():
        logger.info(f"【{func_name}】节点执行完成，没有图片文件夹")
        return state
    #  2.识别md中使用过的图片
    #     参数：md_content,图片路径地址
    #     返回值：[（图片名，图片地址，（上文，下文））]
    targets = step_2_scan_images(md_content,images_dir_obj)

    #  3.进行图片内容的处总结和处理
    #     参数：[（图片名，图片地址，（上文，下文））]，md的文件名称
    #     返回值：[（图片名，图片描述）]
    #  4.上传图片到minIO，更新md的内容
    #     参数：minio_client,{图片名：图片描述}，[（图片名，图片地址，（上文，下文））]，md_content,md文件名称
    #     返回值：更新后的md_content
    #  5.进行数据的处理和备份
    #     参数：new_md_content,md_path,
    #     返回值：new_md_path
    add_done_task(state["task_id"], func_name)
    logger.info(f"【{func_name}】节点执行完成")
    return state