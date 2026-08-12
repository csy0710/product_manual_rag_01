# import os
# import sys
# import json
# import asyncio
# from itertools import count
#
# from app.utils.task_utils import add_done_task, add_running_task
# from app.conf.bailian_mcp_config import mcp_config
# from agents.mcp import MCPServerSse, MCPServerStreamableHttp
# from app.core.logger import logger
# DASHSCOPE_BASE_URL_STREAMBLE = mcp_config.mcp_base_url
# DASHSCOPE_API_KEY = mcp_config.api_key
#
# async def mcp_call_streamable(query):
#     """
#     调用streamable mcp
#     :param query: 查询问题
#     :return:
#     """
#     #创建mcp对象
#     search_mcp = MCPServerStreamableHttp(
#         name="search_mcp",
#         params={
#             "url": DASHSCOPE_BASE_URL_STREAMBLE,
#             "headers": {
#                 "Authorization": f"Bearer {DASHSCOPE_API_KEY}"
#             },
#             "timeout": 10
#         },
#         max_retry_attempts=3
#     )
#     # 连接调用关闭
#     try:
#         # 连接
#         await search_mcp.connect()
#         #获取工具列表
#         tools = await search_mcp.list_tools()
#         logger.info(f"工具列表: {tools}")
#
#         # 调用
#         result = await search_mcp.call_tool(
#             tool_name="bailian_web_search",
#             arguments={
#                 "query": query,
#                 "count": 5
#             }
#         )
#         return result
#     finally:
#         await search_mcp.cleanup()
#
#         # 关闭连接
#
#
# def node_web_search_mcp(state):
#     """
#     节点功能，调用外部搜索引擎补充信息
#     :param state:
#     :return:
#     """
#     add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
#     print("---node-web-search-mcp处理---")
#     #获取问题
#     query = state.get("rewritten_query")
#     #调用streamable mcp
#     result = asyncio.run(mcp_call_streamable(query))
#     # {
#     #     "isError": false,
#     #     "content": [
#     #         {
#     #             "text": "{\"pages\":[{\"snippet\":\"北京时 2026-08-02 07:30更新 | 数据来源中央气象台 今天 周末 7天 8-15天 今天白天 2日(周日) 晴 17 ℃ <3级 日出 06:46 今天夜间 2日(周日) 晴 6 ℃ <3级 日落 17:41 明天白天 3日(周一) 晴 18 ℃ <3级 日出 06:45 逐3小时预报 08时 11时 14时 17时 20时 23时 02时 05时 东南风 <3级 西南风 <3级 西南风 <3级 西南风 <3级 南风 <3级 南风 <3级 南风 <3级 东南风 <3级 8°c 14°c 17°c 16°c 11°c 9°c 8°c 6°c 周边地区 巴特沃斯 19 / 8\",\"hostname\":\"天气网\",\"hostlogo\":\"https://img.alicdn.com/imgextra/i2/O1CN01MJh8gM28s6MYPHsm6_!!6000000007987-2-tps-32-32.png\",\"title\":\"【中央天气】中央今天天气预报,今天,今天天气,7天,15天天气预报,天气预报一周,天气预报15天查询\",\"url\":\"https://www.weather.com.cn/weather1dn/302050100.shtml\"},{\"snippet\":\"11:30更新 | 数据来源中央气象台 今天 7天 8-15天 40天 雷达图 10日(今天) 多云转阴 33 / 24℃ <3级 11日(明天) 雷阵雨 32 / 23℃ <3级 12日(后天) 大雨 27 / 21℃ <3级 13日(周四) 中雨转小雨 27 / 21℃ <3级 14日(周五) 小雨 29 / 22℃ <3级 15日(周六) 小雨 30 / 22℃ <3级 16日(周日) 小雨 30 / 22℃ <3级 分时段预报 生活指数 蓝天预报 11时 14时 17时 20时 23时 02时 05时 08时 31℃ 33℃\",\"hostname\":\"天气网\",\"hostlogo\":\"https://img.alicdn.com/imgextra/i2/O1CN01MJh8gM28s6MYPHsm6_!!6000000007987-2-tps-32-32.png\",\"title\":\"北京天气预报,北京7天天气预报,北京15天天气预报,北京天气查询\",\"url\":\"https://www.weather.com.cn/weather/101010100.shtml\"}],\"request_id\":\"9b82a442-49b0-4731-b040-e74314b32aef\",\"tools\":[{\"result\":\"中国北京市2026-08-11基本天气信息：\\n天气：晴\\n温度：31\\n风向：南风\\n风力：2级\\n近期天气信息：2026-08-10，天气：多云转阴、温度：24/33℃、风向：南风;2026-08-11，天气：雷阵雨、温度：23/32℃、风向：南风转东南风;2026-08-12，天气：大雨、温度：21/27℃、风向：东北风;2026-08-13，天气：中雨转雷阵雨、温度：20/26℃、风向：东北风转北风;2026-08-14，天气：雷阵雨转多云、温度：21/31℃、风向：东南风;2026-08-15，天气：雷阵雨、温度：21/31℃、风向：东南风;2026-08-16，天气：多云转雷阵雨、温度：19/30℃、风向：南风转西风;\",\"type\":\"weather\"}],\"status\":0}",
#     #             "type": "text"
#     #         }
#     #     ]
#     # }
#     # web_documents = []
#     web_documents = json.loads(result.content[0].text).get("pages",[])
#     logger.info(f"mcp搜索的结果为web_documents: {web_documents}")
#
#     add_done_task(state["session_id"],sys._getframe().f_code.co_name,state["is_stream"])
#     # 调用mcp外部引擎
#     print(f"调用外部mcp引擎")
#
#     print("---node-web-search-mcp处理结束---")
#
#     return {"web_search_docs": web_documents}
#
#
# if __name__ == '__main__':
#     # 测试代码：单独运行该文件时，验证MCP搜索功能是否正常
#     print("\n" + "=" * 50)
#     print(">>> 启动 node_web_search_mcp 本地测试")
#     print("=" * 50)
#
#     test_state = {
#         "session_id": "test_mcp_session",
#         "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部 50 mm–170 mm 的局部区域，应在操作面板上如何设置",
#         "is_stream": False
#     }
#
#     try:
#         # 调用MCP搜索节点函数，执行测试
#         result_state = node_web_search_mcp(test_state)
#
#         print("\n" + "=" * 50)
#         print(">>> 测试结果摘要:")
#         search_results = result_state.get('web_search_docs', [])
#         print(f"搜索结果数量: {len(search_results)}")
#         if search_results:
#             print("首条结果预览:")
#             print(json.dumps(search_results[0], indent=2, ensure_ascii=False))
#         else:
#             print("未获取到搜索结果")
#         print("=" * 50)
#
#     except Exception as e:
#         logger.exception(f"测试运行期间发生未捕获异常: {e}")


















import os
import sys
import json
import asyncio
import httpx
import uuid
from app.utils.task_utils import add_done_task, add_running_task
from app.conf.bailian_mcp_config import mcp_config
from app.core.logger import logger

DASHSCOPE_BASE_URL_STREAMBLE = mcp_config.mcp_base_url
DASHSCOPE_API_KEY = mcp_config.api_key

# ------------------------------------------------------------
# 手动 MCP 客户端，避免使用不兼容的 streamable_http_client
# ------------------------------------------------------------
class SimpleBailianMCPClient:
    """直接通过 httpx 发送 JSON-RPC 与 MCP 服务通信"""
    def __init__(self, url, api_key, timeout=10):
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.client = None
        self._session_id = None

    async def connect(self):
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout))
        # 1. 初始化
        init_result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "my-agent", "version": "1.0"}
        })
        # 2. 发送 initialized 通知（某些服务器必须）
        await self._send_notification("notifications/initialized", {})
        return init_result

    async def list_tools(self):
        return await self._send_request("tools/list")

    async def call_tool(self, tool_name, arguments):
        return await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })

    async def _send_request(self, method, params=None):
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": str(uuid.uuid4()),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = await self.client.post(self.url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise Exception(f"MCP 调用错误: {data['error']}")
        return data.get("result")

    async def _send_notification(self, method, params):
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        await self.client.post(self.url, headers=headers, json=payload)

    async def cleanup(self):
        if self.client:
            await self.client.aclose()

async def mcp_call_streamable(query):
    """
    调用streamable mcp（已改为手动 httpx 实现）
    :param query: 查询问题
    :return: 原始调用结果字典
    """
    client = SimpleBailianMCPClient(
        url=DASHSCOPE_BASE_URL_STREAMBLE,
        api_key=DASHSCOPE_API_KEY,
        timeout=10
    )
    try:
        await client.connect()
        logger.info("MCP 连接成功")

        tools = await client.list_tools()
        logger.info(f"工具列表: {tools}")

        raw_result = await client.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": query,
                "count": 5
            }
        )
        # 直接返回原始字典，无需包装
        return raw_result
    finally:
        await client.cleanup()


def node_web_search_mcp(state):
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    print("---node-web-search-mcp处理---")
    query = state.get("rewritten_query")
    web_documents = []

    try:
        result = asyncio.run(mcp_call_streamable(query))
        logger.info(f"MCP 原始返回: {result}")

        # 从返回的结构中提取 pages
        # 格式：{"content": [{"text": "{...}", "type": "text"}], "isError": false}
        if result and "content" in result and len(result["content"]) > 0:
            text = result["content"][0].get("text", "")
            if text:
                data = json.loads(text)
                web_documents = data.get("pages", [])
                logger.info(f"解析到 {len(web_documents)} 条搜索结果")
            else:
                logger.warning("content[0].text 为空")
        else:
            logger.warning("MCP 返回结果格式异常")
    except Exception as e:
        logger.error(f"MCP 搜索调用失败: {e}")

    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    print("---node-web-search-mcp处理结束---")
    return {"web_search_docs": web_documents}

if __name__ == '__main__':
    # 测试代码保持不变
    print("\n" + "=" * 50)
    print(">>> 启动 node_web_search_mcp 本地测试")
    print("=" * 50)

    test_state = {
        "session_id": "test_mcp_session",
        "rewritten_query": "华为手机怎么样",
        "is_stream": False
    }

    try:
        result_state = node_web_search_mcp(test_state)
        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        search_results = result_state.get('web_search_docs', [])
        print(f"搜索结果数量: {len(search_results)}")
        if search_results:
            print("首条结果预览:")
            print(json.dumps(search_results[0], indent=2, ensure_ascii=False))
        else:
            print("未获取到搜索结果")
        print("=" * 50)
    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")