import os
import sys
from pathlib import Path
from typing import Tuple, List

from aiohttp.web_fileresponse import content_type

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
        with md_path_obj.open("r",encoding="utf-8") as f:
            state["md_content"] = f.read()
            logger.info(f"【step_1_get_content】节点执行完成，读取文件内容，文件路径：{md_path_obj}")



    # 图片路径是否为空
    # 自己传入的md文件中的图片文件夹也要叫images，这里没有特殊处理
    images_dir_obj =md_path_obj.parent / "images"

    return state["md_content"],md_path_obj,images_dir_obj


def find_image_in_md_content(md_content, image_file,context_length:int=100):
    """在md中查找图片上下文"""
    # 定义正则表达式
    # pattern = re.compile(r"!\[.*?\(.*?"+image_file+".*?\)")
    pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r"\)")
    content = None #存储图片多次使用，上下文不同！minerU解析的图片即使两个图片一样但是图片名称不同，也会被识别为不同的图片！
    # 查询符合位置
    items = list(pattern.finditer(md_content))
    if not items:
        return None
    if item := items[0]:
        start,end = item.span()#获取的是匹配的图片链接的起始位置和结束位置
        #截取上文
        pre_text = md_content[max(start-context_length,0):start]
        #截取下文
        post_text = md_content[end:min(end+context_length,len(md_content))]
        content=((pre_text,post_text))
    # 截取位置前后内容
    if content:
        logger.info(f"【find_image_in_md_content】节点执行完成，图片 {image_file} 在md中使用，截取第一个上下文为：{content}")
        return content


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
    logger.info(f"【step_2_scan_images】节点执行完成，共识别到 {len(targets)} 张图片")

    return targets


def step_3_generate_img_summaries(targets, stem):
    """生成图片描述
    参数：[（图片名，图片地址，（上文，下文））]，md的文件名称
    返回值：[（图片名，图片描述）]
    """
    # 循环每一张图片，向视觉模型进行请求，获取图片描述
    summaries = {}
    requests_times=deque()
    for image_file,image_path,context in targets:

        # 访问限速（模型的限速标准1分钟10次）滑动窗口
        apply_api_rate_limit(requests_times,max_requests=9)


        vm_model = get_llm_client(model=lm_config.lv_model)
        prompt = load_prompt("image_summary",root_folder=stem,image_content=context)
        with open(image_path,"rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        messages = [
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": prompt},
                ]
            )
        ]
        response = vm_model.invoke(messages)
        summary= response.content.strip().replace("\n","")
        summaries[image_file] = summary
    logger.info(f"【step_3_generate_img_summaries】节点执行完成,共生成 {len(summaries)} 张图片的描述,图片描述为：{summaries}")
    return summaries


def step_4_upload_images_and_replace_md(summaries, targets, md_content, stem):
    """上传图片到minIO，更新md的内容"""
    # 获取minio客户端对象
    minio_client = get_minio_client()
    images_url={}
    # 删除旧的图片
    object_list = minio_client.list_objects(minio_config.bucket_name, prefix=f"{minio_config.minio_img_dir[1:]}/{stem}",recursive=True)
    delete_object_list = [DeleteObject(name=obj.object_name) for obj in object_list]

    errors = minio_client.remove_objects(minio_config.bucket_name,delete_object_list)
    for error in errors:
        logger.info(f"删除对象失败,错误信息为：{error}")
    logger.info(f"已经完成了{stem}中的删除,共删除 {len(delete_object_list)} 张图片")

    # 上传新的图片到minIO.targets中的数据格式为（图片名.jpg，图片地址，（上文，下文））
    for image_file,image_path,_ in targets:
        try:
            minio_client.fput_object(
                bucket_name = minio_config.bucket_name,
                object_name = f"{minio_config.minio_img_dir[1:]}/{stem}/{image_file}",
                file_path=image_path,
                content_type ="image/jpeg"
            )
            # 得到传完后的字典，key为图片名，value为minio图片地址
            images_url[image_file] = f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_file}"
            logger.info(f"【step_4_upload_images_and_replace_md】节点执行完成，图片 {image_file} 上传到minIO成功,图片地址为：{images_url[image_file]}")
        except Exception as e:
            logger.error(f"【step_4_upload_images_and_replace_md】节点执行完成，图片 {image_file} 上传到minIO失败,错误信息为：{e}")
    image_infos = {}
    for image_file,summary in summaries.items():
        if url := images_url.get(image_file):
            image_infos[image_file] = (summary,url)
    logger.info(f"图片处理的汇总结果:{image_infos}")
    # 在md_content中进行匹配，替换为图片描述和图片地址
    if image_infos:
        for image_file,(summary,url) in image_infos.items():
            rep = re.compile(r"!\[.*?]\(.*?"+image_file+r".*?\)")
            md_content = rep.sub(f"![{summary}]({url})",md_content)
        logger.info(f"已经完成md内容的替换，新的内容为:{md_content}")
    return md_content


def step_5_backup_md(new_md_content, md_path_obj):
    """完成新的md的磁盘备份，并返回老地址"""
    # 将地址分割例如：c:/xxx/xxx.md -> c:/xxx/xxx_new.md
    new_md_path_str = os.path.splitext(md_path_obj)[0]+"_new.md"
    with open(new_md_path_str,"w",encoding="utf-8") as f:
        f.write(new_md_content)
    return new_md_path_str


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
    summaries = step_3_generate_img_summaries(targets,md_path_obj.stem)

    #  4.上传图片到minIO，更新md的内容
    #     参数：minio_client,{图片名：图片描述}，[（图片名，图片地址，（上文，下文））]，md_content,md文件名称
    #     返回值：更新后的md_content
    new_md_content = step_4_upload_images_and_replace_md(summaries,targets,md_content,md_path_obj.stem)

    #  5.进行数据的处理和备份
    #     参数：new_md_content,md_path,
    #     返回值：new_md_path
    new_md_file_path = step_5_backup_md(new_md_content,md_path_obj)

    # 更新state中的md_path
    state["md_path"] = new_md_file_path

    # 更新state中的md_content
    state["md_content"] = new_md_content

    add_done_task(state["task_id"], func_name)
    logger.info(f"【{func_name}】节点执行完成")
    return state



if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\20260808\58681287-0e20-41e9-8c08-d80831d84967\hak180产品安全手册", "hak180产品安全手册.md")
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
            "md_content": ""
        }
        logger.info("开始本地测试 - MD图片处理全流程")
        # 执行核心处理流程
        result_state = node_md_img(test_state)
        logger.info(f"本地测试完成 - 处理结果状态：{result_state}")