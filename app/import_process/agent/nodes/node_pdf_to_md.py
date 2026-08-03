import os
import shutil
import sys
import time
import zipfile
from app.import_process.agent.state import ImportGraphState,create_default_state
from pathlib import Path
import requests
from pandas.core.groupby.ops import extract_result
import dotenv
from app.utils.format_utils import format_state
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import add_running_task
from app.utils.task_utils import add_running_task, add_done_task
from app.conf.mineru_config import mineru_config, MineruConfig





"""
   node_pdf_to_md
   参数：state[is_pdf_read_enabled = True，pdf_path,local_dir]
   返回值：state[md_path]
   1.打印日志参数
   2.调用step_1_validate_paths，校验文件路径参数以及输出文件地址是否存在。文件存在校验。
   3.调用step_2_upload_and_poll，minerU的交互
   4.调用step_3_download_and_extract，下载转换后的md压缩文件并且解压到目标文件夹下。
   5.打印结束日志参数
   6.返回md_path
   
   
   step_1_validate_paths
   参数：pdf_path,local_dir
   返回值:pdf_path_obj,local_dir_obj
   1.校验pdf文件路径是否存在
   2.校验输出文件地址是否存在
   3.使用.exists（）方法校验文件是否存在
   4.校验输出地址文件夹是否存在，如果不存在，就创建一个文件夹
   5.返回校验完毕可以直接使用的路径对象。返回参数pdf的文件路径，输出文件地址。
   
   
   step_2_upload_and_poll
   参数：pdf_path_obj
   返回值：pdf转md文件的zip包下载地址
   1.按照minerU中的配置，配置请求头，发送post请求，上传pdf文件
   2.判断返回的response中的状态是否为200，如果不是，就抛出异常
   3.在返回的response中，提取下载地址，以及任务id
   4.创建session对象，禁止走代理，避免因为代理导致请求头被修改。
   5.使用try-except语句，捕获可能的异常，如网络错误，超时等。with语句读到pdf文件内容，发送post请求，上传pdf文件
   6.判断返回的response中的状态是否为200，如果不是，就抛出异常
   7.轮询minerU的api，查询任务状态，直到任务完成
   8.返回下载地址
   
   
   
   step_3_download_and_extract
   参数：zip包下载地址,local_dir_obj
   返回值：解压后的文件路径
   1.使用requests库，发送get请求，下载zip包
   2.判断返回的response中的状态是否为200，如果不是，就抛出异常
   3.使用zipfile库，解压zip包到目标文件夹
   4.返回解压后的文件路径
"""





def step_1_validate_paths(state: ImportGraphState):
    """
    校验参数，pdf_path如果失效，直接进行异常处理
    local_dir如果没有，就创建一个
    返回校验完毕可以直接使用的路径对象。返回参数pdf的文件路径，输出文件地址。
    """
    logger.debug(f"[step_1_validate_paths]在md转pdf中开始进行参数校验。")
    pdf_path = state["pdf_path"]
    local_dir = state["local_dir"]

    #常规的非空校验(校验文件路径是否存在)
    if not pdf_path:
        logger.error(f"[step_1_validate_paths]检查发现没有输出文件，无法继续解析")
        raise ValueError(f"[step_1_validate_paths]检查发现没有输出文件，无法继续解析")

    if not local_dir:
        local_dir = PROJECT_ROOT/"output"
        logger.info(f"[step_1_validate_paths]检查发现没有输出文件地址，默认使用{local_dir}")

    #文件是否存在校验
    pdf_path_obj = Path(pdf_path)
    local_dir_obj = Path(local_dir)
    # 校验pdf文件是否存在
    if not pdf_path_obj.exists():
        logger.error(f"[step_1_validate_paths]pdf_path不存在，请检查文件路径是否正确")
        raise FileNotFoundError(f"[step_1_validate_paths]pdf_path不存在，请检查文件路径是否正确")

    if not local_dir_obj.exists():
        logger.error(f"[step_1_validate_paths]local_dir不存在，主动创建对应文件夹")
        local_dir_obj.mkdir(parents=True, exist_ok=True)

    return pdf_path_obj,local_dir_obj


def step_2_upload_and_poll(pdf_path_obj)->str:
    """
    调用MinerU (magic-pdf) 工具，上传pdf文件，返回解析后的zip包下载地址。
    :param pdf_path_obj: pdf文件路径对象
    :return: zip包下载地址
    """
    logger.debug(f"[step_2_upload_and_poll]在md转pdf中开始进行参数校验。")
#    1.申请上传解析的地址

    # dotenv.load_dotenv()
    # 请求头数据
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"


    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }


    data = {
        "files": [
            {"name": f"{pdf_path_obj}"}
        ],
        "model_version": "vlm"
    }

    response = requests.post(url, headers=header, json=data)

    # 结果处理 请求http状态码不是200，或者返回的code不是0，都说明请求失败
    if response.status_code != 200 or response.json()["code"] !=0:
        logger.error(f"[step_2_upload_and_poll]调用MinerU (magic-pdf) 请求上传连接失败，错误信息：{response.json()}")
        raise RuntimeError(f"[step_2_upload_and_poll]调用MinerU (magic-pdf) 请求上传连接失败，错误信息：{response.json()}")
    uploaded_url = response.json()["data"]['file_urls'][0]#在这个地址上传文件
    batch_id = response.json()["data"]['batch_id']#处理id，根据这个id获取结果


    #2.将文件上传到对应的解析地址
    #使用put请求将pdf_path_obj文件传递到uploaded_url地址。不能直接使用put，很大概率报错
    #get/post宽进宽出 put严进严出。
    http_session = requests.Session()
    http_session.trust_env = False#禁止走代理


    try:
        with open(pdf_path_obj,'rb')as f:
            file_data = f.read()

        upload_response = http_session.put(uploaded_url,data=file_data)
        if upload_response.status_code !=200:
            logger.error(f"[step_2_upload_and_poll]调用MinerU (magic-pdf) 获取请求数据失败，错误信息：{upload_response.json()}")
            raise RuntimeError(f"[step_2_upload_and_poll]调用MinerU (magic-pdf) 获取请求数据失败，错误信息：{upload_response.json()}")
    except Exception as e:
        logger.error(f"[step_2_upload_and_poll]调用MinerU (magic-pdf) 获取请求数据失败，错误信息：{e}")
        raise RuntimeError(f"[step_2_upload_and_poll]调用MinerU (magic-pdf) 获取请求数据失败，错误信息：{e}")
    finally:
        http_session.close()


    #3.轮询获取解析结果
    url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout_seconds = 600
    poll_interval = 3
    start_time = time.time()#进入的起始时间
    while True:
    #超时判断
        if time.time() - start_time > timeout_seconds:
            logger.error(f"[step_2_upload_and_poll]调用MinerU (magic-pdf) 工具上传文件失败，超时时间超过{timeout_seconds}秒")
            raise RuntimeError(f"[step_2_upload_and_poll]调用MinerU (magic-pdf) 工具上传文件失败，超时时间超过{timeout_seconds}秒")
        #向指定的url地址获取本次解析的结果
        res = requests.get(url,headers=header)

        #解析结果判断和获取zip——url
        if res.status_code!=200:
            #如果报的错误是5xx,我们会给予机会，直到timeout
            if 500<=res.status_code<600:
                time.sleep(poll_interval)
                continue
            raise RuntimeError(f"[step_2_upload_and_poll]调用MinerU (magic-pdf) 工具上传文件失败，返回的状态码：{res.status_code}")

        json_data = res.json()#获取本次结果
        if json_data["code"]!=0:
            #code!=0 很大概率token过期了
            logger.error(f"[step_2_upload_and_poll]调用MinerU (magic-pdf) 工具上传文件失败，错误信息：{json_data['msg']}")
            raise RuntimeError(f"[step_2_upload_and_poll]调用MinerU (magic-pdf) 工具上传文件失败，错误信息：{json_data['msg']}")
        extract_result = json_data["data"]["extract_result"][0]

        if extract_result['state'] == 'done':
            #解析完毕获取结果
            full_zip_url = extract_result['full_zip_url']
            logger.info(f"pdf解析完成，耗时{time.time() - start_time}秒，解析结果：{full_zip_url}")
            return full_zip_url
        else:
            #解析未完成，继续轮询
            time.sleep(poll_interval)

def step_3_download_and_extract(zip_url,local_dir_obj,stem):
    """
    下载zip包，解压并提取（local_dir）
    参数：zip_url zip包下载地址，local_dir_obj输出文件地址，stem文件名（不包含扩展名）。返回值，解压后的markdown文件地址。
    """
    #1.下载zip包 respons响应体
    response = requests.get(zip_url)
    if response.status_code != 200:
        logger.error(f"[step_3_download_and_extract]调用MinerU (magic-pdf) 请求下载连接失败，错误信息：{response.json()}")
        raise RuntimeError(f"[step_3_download_and_extract]调用MinerU (magic-pdf) 请求下载连接失败，错误信息：{response.json()}")
    # 2.将响应体的zip文件保存到本地
    zip_save_path = local_dir_obj / f"{stem}_result.zip"
    with open(zip_save_path,'wb')as f:
        f.write(response.content)
    logger.info(f"zip包下载完成，保存路径：{zip_save_path}")
    #3.清空旧目录（将上一次的解压结果删除）
    #创建解压目录
    extract_target_dir = local_dir_obj / stem
    # 递归删除旧目录
    if extract_target_dir.exists():
        shutil.rmtree(extract_target_dir)
    #创建一个新目录
    extract_target_dir.mkdir(parents=True,exist_ok=True)
    #4.解压zip文件到local_dir_obj地址
    #创建一个zip文件对象，用于解压
    with zipfile.ZipFile(zip_save_path,'r') as zip_file_obj:
    # 调用对象的方法，解压所有文件到指定目录 参数：解压的目录
        zip_file_obj.extractall(extract_target_dir)
    #5.返回解压后的markdown文件地址
    md_file_list = list(extract_target_dir.rglob('*.md'))
    logger.info(f"解压后的markdown文件列表：{md_file_list}")

    # 判断list是否为空
    if not md_file_list:
        raise RuntimeError(f"[step_3_download_and_extract]解压后的markdown文件列表为空，无法继续执行")

    #检查是否有与pdf同名的markdown文件
    target_md_file = None
    for md_file in md_file_list:
        if md_file.name == f"{stem}.md":
            target_md_file = md_file
            logger.info(f"找到与pdf同名的markdown文件：{target_md_file}")


    #如果没有找到与pdf同名的markdown文件，检查是否有默认full.md的markdown文件
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                break
    logger.info(f"找到默认full.md的markdown文件：{target_md_file}")


    #如果都没有找到，使用第一个markdown文件
    if not target_md_file:
        target_md_file = md_file_list[0]
    logger.info(f"使用第一个markdown文件：{target_md_file}")

    #将不同的文件名统一名称
    if target_md_file!=stem:
        target_md_file = target_md_file.rename(target_md_file.with_name(f"{stem}.md"))
        logger.info(f"统一文件名：{target_md_file}")


    #将路径转换为字符串
    # 解析路径，确保路径是绝对路径
    target_file_str =str(target_md_file.resolve())
    logger.info(f"完成md解压，返回的markdown文件地址：{target_file_str}")
    return target_file_str

def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1. 调用 MinerU (magic-pdf) 工具。
    2. 将 PDF 转换成 Markdown 格式。
    3. 将结果保存到 state["md_content"]。
    """
    # 1.日志输出
    # 获取当前函数名称
    func_name = sys._getframe().f_code.co_name
    logger.info(f"节点【{func_name}】启动,当前工作流状态：{format_state(state)}")
    add_running_task(state["task_id"],func_name)
    try:
    # 2.进行参数校验（local_dir|local_file_path完成字面意思的校验，校验文件是否真的存在）
    # 返回校验完毕可以直接使用的路径对象。返回参数pdf的文件路径，输出文件地址。
        pdf_path_obj,local_dir_obj = step_1_validate_paths(state)
    # 3.调用MinerU (magic-pdf) 工具，进行pd解析（文件local_file_path）返回一个下载文件的地址
    # 参数：要解析的pdf文件路径。 返回值：解析后的zip包下载地址。
        zip_url = step_2_upload_and_poll(pdf_path_obj)
    # 4.下载zip包，解压并提取（local_dir）
    # 参数：zip包下载地址，输出文件地址，文件名（不包含扩展名）。返回值，解压后的markdown文件地址。
        md_path = step_3_download_and_extract(zip_url,local_dir_obj,pdf_path_obj.stem)
    # 5.更新state，把md_path地址进行赋值，读取md的文件内容 md_content赋值（文本内容）
        state['md_path'] = md_path
        state['local_dir'] = str(local_dir_obj)
    # md的内容读取
        with open(md_path,'r',encoding='utf-8') as f:
            state['md_content'] = f.read()
    # 6.结束日志和任务状态的配置
    except Exception as e:
        # 处理异常
        logger.error(f"【{func_name}】在使用MinerU (magic-pdf) 工具解析PDF文件时，发生异常信息：{e}")
        raise e #中支工作流
    finally:
        # 结束，记录节点运行状态打印当前工作流状态
        add_done_task(state["task_id"], func_name)
        logger.info(f"【{func_name}】节点执行完成，\n更新后工作流状态：{format_state(state)}")
    logger.info(f">>> [Stub] 执行节点: {sys._getframe().f_code.co_name}")



    return state




if __name__ == "__main__":

    # 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")

    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")

    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)

    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
    )

    node_pdf_to_md(test_state)

    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")