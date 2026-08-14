from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from app.import_process.agent.state import create_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import *
from app.core.logger import logger
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.query_process.agent.main_graph import query_app

# 后续导入启动图对象
#from app.query_process.main_graph import query_app


# 定义fastapi对象
app = FastAPI(title="query service",description="掌柜智库查询服务！")
# 跨域问题解决
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


#6个接口  健康状态 返回页面 发起提问 sse长连接 查看历史对话 清空历史对话

@app.get("/health")
async def health():
    logger.info("触发后台健康检查接口，数据正常")
    return {"status": "ok"}



#返回chat页面
@app.get("/chat.html")
async def chat_html():

    #查找html页面的地址
    chat_html_path = PROJECT_ROOT /"app" / "query_process" / "page" / "chat.html"
    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail="HTML文件不存在")

    return FileResponse(chat_html_path)


#发起提问
#定义一个接收参数的类型
class QueryRequest(BaseModel):
    query: str = Field(...,title="查询内容，必须传递")
    session_id: str = Field(None,title="会话ID，可以不传递，后台uuid生成第一个")
    is_stream: bool = Field(False,title="是否是否流式返回，默认False")

def run_query_graph(session_id: str, query: str, is_stream: bool):
    """
    运行查询图
    :param session_id: 会话ID
    :param query:
    :param is_stream:
    :return:
    """
    # 本次任务开启了流式输出 is_stream 为True，把结果加入到sse队列中
    update_task_status(session_id, "processing",is_stream)
    state =create_default_state(
        session_id = session_id,
        original_query = query,
        is_stream = is_stream,
    )
    try:
        query_app.invoke(state)
        update_task_status(session_id, "completed", is_stream)
    except Exception as e:
        logger.error(f"查询图执行异常，会话ID为{session_id}，异常信息为{e}")
        logger.exception(f"异常信息为{e}")
        update_task_status(session_id, "failed", is_stream)
        push_to_session(session_id,SSEEvent.ERROR,{"error":str(e)})





@app.post("/query")
async def query(request: QueryRequest,background_tasks: BackgroundTasks):
    """
    发起提问
    :param request: 查询参数
    :param background_tasks:
    :return:
    """
    query = request.query
    session_id = request.session_id
    is_stream = request.is_stream

    #判断是不是流式输出
    if is_stream:
        #创建一个sse队列
        create_sse_queue(session_id)

        background_tasks.add_task(run_query_graph, session_id, query,is_stream)
        logger.info(f"开启流式输出，会话ID为{session_id}")
        return {"session_id": session_id,
                "messages":"本次查询处理中"}
    else:
        #同步执行
        run_query_graph(session_id, query,is_stream)
        #获取最后一个节点插入的结果
        answer = get_task_result(session_id,"answer")
        logger.info(f"query:{query}开启同步查询，结果为{answer}")
        #返回对应的json数据
        return {"answer": answer,
                "session_id": session_id,
                "messages":"本次查询处理完成",
                "done_list":[]
                }



@app.get("/stream/{session_id}")
async def stream(session_id: str,request: Request):
    """
    :param session_id: 会话ID
    :param request:前端的原生请求对象，可以判断用户是否关闭了连接
    :return:
    """
    logger.info(f"session_id = {session_id}客户端连接成功")
    return StreamingResponse(
        sse_generator(session_id,request),
        media_type="text/event-stream"
    )


@app.get("/history")
async def history(session_id: str,limit: int = 10):
    """
    查看历史对话
    :param session_id: 会话ID
    :param limit: 最多返回limit条数据
    :return:
    """
    # 从数据库中获取历史对话
    chats = get_recent_messages(session_id,limit)
    #chat 在mongodb中有一个ObjectId字段，不能直接序列化，需要转换为字符串类型

    # items = []
    # for chat in chats:
    #     items.append(chat)
    logger.info(f"查看历史对话，会话ID为{session_id},查询数据为{chats}")
    # 返回历史对话
    return {"session_id": session_id,
            "items": chats}

@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """
    清空历史对话
    """
    delete_count = clear_history(session_id)
    logger.info(f"清空历史对话，会话ID为{session_id},删除{delete_count}条数据")
    return {"delete_count": delete_count,
            "messages":f"{delete_count}历史对话已清空"
            }
    # 清空历史对话

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)







