# 掌柜智库 · 面向产品手册的 RAG 知识库问答系统

一个基于 **LangGraph** 搭建的双流程 RAG 系统：把 PDF / Markdown 产品手册灌入知识库，再面向用户提供带多路召回、融合排序、可追溯引用的问答服务。

系统按「导入」与「查询」拆成两个互相独立的服务，各自是一条 LangGraph 状态图，通过 Milvus / MongoDB / MinIO 三个存储层解耦。

---

## 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [API 说明](#api-说明)
- [核心设计说明](#核心设计说明)
- [测试](#测试)
- [已知限制](#已知限制)

---

## 功能特性

**导入侧**

- PDF 自动解析为 Markdown（调用 MinerU 云端 API），图片自动提取并上传 MinIO，Markdown 中的图片链接自动改写
- 基于标题层级的**语义分块**：先按 Markdown 标题切 section，再对超长 section 做二次切分，并对同一父标题下的碎片块做合并
- **主体名称识别**：用大模型从文档开头切片中提取产品名（如「华为 MateBook B3-420」），作为后续检索的硬过滤条件
- **BGE-M3 稠密 + 稀疏混合向量化**，一次入库同时写入两种向量
- 后台异步执行 + 任务状态实时轮询

**查询侧**

- **主体确认节点**：先识别用户问题里的产品名，在主体库中做混合检索并按分数分段（≥0.85 确认 / ≥0.6 候选）。命中不了确认集合时不硬答，而是返回「您想咨询哪个商品」的澄清话术
- **三路并发召回**：普通向量检索、HyDE 检索、联网检索（百炼 WebSearch MCP）
- **加权 RRF 融合**：不同召回源可配不同权重，按 `weight / (60 + rank)` 累加得分
- **Rerank + 断崖截断**：bge-reranker-large 精排后，按相对/绝对分差动态决定 TopK，避免把低质量切片塞进上下文
- **SSE 流式输出**：逐 token 推送给前端，同时推送节点级进度事件
- 多轮对话历史落 MongoDB，支持历史查询与清空

---

## 系统架构

### 导入流程

```
                    ┌──────────────┐
  上传 PDF/MD ─────▶ │  node_entry  │  判断文件类型，置位 is_pdf_read_enabled / is_md_read_enabled
                    └──────┬───────┘
                           │ (条件边)
              ┌────────────┴────────────┐
              ▼                         ▼
      ┌───────────────┐          ┌──────────────┐
      │ node_pdf_to_md│          │ node_md_img  │  ← MD 文件直接进这里
      └───────┬───────┘          └──────┬───────┘
              │  MinerU API             │  图片上传 MinIO + 链接改写
              └───────────┬─────────────┘
                          ▼
                 ┌──────────────────┐
                 │node_document_split│  标题语义切分 + 二次切分 + 短块合并
                 └────────┬─────────┘
                          ▼
            ┌───────────────────────────┐
            │ node_item_name_recognition│  LLM 提取产品主体名 → kb_item_names
            └────────────┬──────────────┘
                         ▼
              ┌─────────────────────┐
              │  node_bge_embedding │  BGE-M3 生成稠密 + 稀疏向量
              └──────────┬──────────┘
                         ▼
              ┌─────────────────────┐
              │  node_import_milvus │  写入 kb_chunks
              └──────────┬──────────┘
                         ▼
                        END
```

### 查询流程

```
 用户提问 ──▶ ┌──────────────────────────┐
              │ node_item_name_confirm   │  识别主体名 → 主体库混合检索 → 分段判定
              └────────────┬─────────────┘
                           │ (条件边)
        ┌──────────────────┴───────────────────┐
        │ 主体不明确：直接给出澄清话术           │ 主体明确：继续检索
        ▼                                      ▼
┌────────────────────┐        ┌───────────────────────────────────────┐
│ node_answer_output │        │ node_search_embedding      (普通向量)  │
└─────────┬──────────┘        │ node_search_embedding_hyde (HyDE)     │
          │                   │ node_web_search_mcp        (联网检索)  │
          │                   └───────────────┬───────────────────────┘
          │                                   ▼
          │                          ┌────────────────┐
          │                          │   node_rrf     │  加权 RRF 融合
          │                          └───────┬────────┘
          │                                  ▼
          │                          ┌────────────────┐
          │                          │  node_rerank   │  精排 + 断崖截断 TopK
          │                          └───────┬────────┘
          │                                  ▼
          │                          ┌────────────────┐
          └─────────────────────────▶│node_answer_output│ LLM 生成 / SSE 流式输出
                                     └────────┬────────┘
                                              ▼
                                             END
```

### 存储与外部依赖

| 组件 | 用途 | 必需 |
| --- | --- | --- |
| Milvus | 切片向量库 `kb_chunks`、主体名库 `kb_item_names` | ✅ |
| MongoDB | 多轮对话历史 | ✅ |
| MinIO | 原始文件与手册内图片 | ✅ |
| MinerU 云端 API | PDF → Markdown 解析 | ✅（仅导入 PDF 时需要） |
| 阿里云百炼 DashScope | qwen-flash 对话、qwen3-vl-flash 图片理解、WebSearch MCP | ✅ |
| BGE-M3（本地权重） | 稠密 + 稀疏混合向量化 | ✅ |
| bge-reranker-large（本地权重） | 检索结果精排 | ✅ |
| Neo4j | 图检索客户端（代码保留，当前主流程未接入） | ❌ |

---

## 技术栈

- **工作流编排**：LangGraph 1.2（StateGraph + 条件边 + 并行扇出）
- **Web 框架**：FastAPI + Uvicorn，SSE 通过 `StreamingResponse` 实现
- **向量检索**：PyMilvus 3.0（`AnnSearchRequest` + `WeightedRanker` 混合检索）
- **大模型接入**：LangChain `ChatOpenAI`（OpenAI 兼容协议，指向百炼 DashScope）
- **本地模型**：BGE-M3（`pymilvus.model.BGEM3EmbeddingFunction`）、bge-reranker-large（`FlagEmbedding.FlagReranker`）
- **存储客户端**：pymilvus / pymongo / minio
- **日志**：loguru（控制台 + 文件双通道，按天切分、自动清理）
- **Python**：3.12

---

## 目录结构

```
dataset_arg_project/
├── app/
│   ├── import_process/              # 【导入侧】PDF/MD → 向量库
│   │   ├── agent/
│   │   │   ├── main_graph.py        # 导入状态图：节点注册与边定义
│   │   │   ├── state.py             # ImportGraphState 状态定义
│   │   │   └── nodes/               # 7 个导入节点
│   │   │       ├── node_entry.py                    # 入口：判断文件类型
│   │   │       ├── node_pdf_to_md.py                # PDF → Markdown（MinerU）
│   │   │       ├── node_md_img.py                   # 图片上传 MinIO + 链接改写
│   │   │       ├── node_document_split.py           # 语义分块
│   │   │       ├── node_item_name_recognition.py    # 主体名识别
│   │   │       ├── node_bge_embedding.py            # 混合向量化
│   │   │       └── node_import_milvus.py            # 入库 Milvus
│   │   ├── api/import_server.py     # 导入服务（:8000）
│   │   └── page/import.html         # 上传页
│   ├── query_process/               # 【查询侧】提问 → 多路召回 → 回答
│   │   ├── agent/
│   │   │   ├── main_graph.py        # 查询状态图
│   │   │   ├── state.py             # QueryGraphState 状态定义
│   │   │   └── nodes/               # 7 个查询节点
│   │   │       ├── node_item_name_confirm.py        # 主体确认 + 问题改写
│   │   │       ├── node_search_embedding.py         # 普通向量检索
│   │   │       ├── node_search_embedding_hyde.py    # HyDE 检索
│   │   │       ├── node_web_search_mcp.py           # 联网检索（MCP）
│   │   │       ├── node_rrf.py                      # 加权 RRF 融合
│   │   │       ├── node_rerank.py                   # 精排 + TopK 截断
│   │   │       └── node_answer_output.py            # 生成答案 / SSE 输出
│   │   ├── api/query_server.py      # 查询服务（:8001）
│   │   ├── sse/sse_step_1.py        # SSE 分步调试脚本
│   │   └── page/chat.html           # 对话页
│   ├── clients/                     # 存储客户端单例封装
│   │   ├── milvus_utils.py          # 连接、建表、混合检索
│   │   ├── mongo_history_utils.py   # 对话历史 CRUD
│   │   ├── minio_utils.py           # 对象存储
│   │   └── neo4j_utils.py           # 图数据库（预留）
│   ├── lm/                          # 模型调用封装
│   │   ├── lm_utils.py              # ChatOpenAI 客户端（带缓存）
│   │   ├── embedding_utils.py       # BGE-M3 单例 + 混合向量生成
│   │   └── reranker_utils.py        # FlagReranker 单例
│   ├── conf/                        # 各组件配置类（统一从 .env 读取）
│   ├── core/                        # 日志、Prompt 模板加载
│   ├── utils/                       # 路径、任务状态、SSE、向量归一化等工具
│   └── test/                        # 调试脚本（非 pytest 用例，会真实调用外部服务）
├── prompts/                         # 提示词模板（与代码解耦）
├── tests/                           # pytest 用例
├── doc/                             # 测试语料 PDF（不入库）
├── output/                          # 导入中间产物（不入库）
└── logs/                            # 运行日志（不入库）
```

---

## 快速开始

### 环境要求

- Python 3.12
- 可用的 Milvus、MongoDB、MinIO 实例
- 阿里云百炼 API Key（对话 / 视觉 / WebSearch MCP 共用）
- MinerU API Token（PDF 解析）
- 本机已下载 BGE-M3 与 bge-reranker-large 权重

### 1. 启动依赖服务

最省事的方式是用 Docker 起 Milvus、MongoDB、MinIO：

```bash
# Milvus（单机版）
# 参考官方文档 https://milvus.io/docs/install_standalone-docker-compose.md
wget https://github.com/milvus-io/milvus/releases/download/v2.4.x/milvus-standalone-docker-compose.yml -O docker-compose.yml
docker compose up -d

# MongoDB
docker run -d --name mongo -p 27017:27017 mongo:7

# MinIO
docker run -d --name minio -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

> MinIO 起来后需要手动创建一个桶（默认名 `knowledge-base-files`），否则图片上传会失败。

### 2. 安装依赖

```bash
conda create -n dataset_arg_project python=3.12 -y
conda activate dataset_arg_project
pip install -r requirements.txt
```

GPU 用户请先按 [PyTorch 官方指引](https://pytorch.org/get-started/locally/) 装好对应 CUDA 版本的 torch，再装其余依赖。

### 3. 下载本地模型

项目需要两个本地权重（或用 `modelscope download` 手动拉取）：

```bash
# BGE-M3 —— 稠密 + 稀疏混合向量化
modelscope download --model BAAI/bge-m3 --local_dir /path/to/models/BAAI/bge-m3

# bge-reranker-large —— 精排
modelscope download --model BAAI/bge-reranker-large \
  --local_dir /path/to/models/rerank/BAAI/bge-reranker-large
```

仓库里的 `app/tool/download_bgem3.py`、`app/tool/download_reranker.py` 也是做这件事的脚本，可按需使用。

### 4. 配置环境变量

```bash
cp .env.example .env
```

然后按注释填写 `.env`。**至少必须改这几项：**

| 变量 | 说明 |
| --- | --- |
| `OPENAI_API_KEY` / `BAILIAN_API_KEY` | 百炼 API Key |
| `MINERU_API_TOKEN` | MinerU Token |
| `BGE_M3_PATH` | BGE-M3 本地权重绝对路径 |
| `BGE_RERANKER_LARGE` | bge-reranker-large 本地权重绝对路径 |
| `MILVUS_URL` / `MONGO_URL` / `MINIO_*` | 三个存储服务的连接信息 |

CPU 环境下务必保持 `BGE_FP16=0`、`BGE_RERANKER_DEVICE=cpu`。

> `.env` 已在 `.gitignore` 中屏蔽，切勿提交真实密钥。

### 5. 启动服务

两个服务需要**分别启动**：

```bash
# 终端 1：导入服务（8000）
python -m app.import_process.api.import_server

# 终端 2：查询服务（8001）
python -m app.query_process.api.query_server
```

### 6. 使用

- 导入手册：浏览器打开 <http://127.0.0.1:8000/import>，上传 PDF / Markdown，页面会轮询展示每个节点的执行进度
- 开始提问：浏览器打开 <http://127.0.0.1:8001/chat.html>

---

## API 说明

### 导入服务（:8000）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/import` | 返回上传页面 `import.html` |
| `POST` | `/upload` | 上传文件（支持多文件），返回 `task_ids`，后台异步执行导入图 |
| `GET` | `/status/{task_id}` | 轮询任务状态，返回已完成节点与运行中节点 |

### 查询服务（:8001）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `GET` | `/chat.html` | 返回对话页面 |
| `POST` | `/query` | 发起提问。`is_stream=false` 直接返回答案，`true` 则先建 SSE 队列再后台执行 |
| `GET` | `/stream/{session_id}` | SSE 长连接，事件类型：`ready` / `progress` / `delta` / `final` / `error` |
| `GET` | `/history?session_id=&limit=` | 查询历史对话 |
| `DELETE` | `/history/{session_id}` | 清空历史对话 |

---

## 核心设计说明

### 1. 为什么先做「主体确认」再检索

通用 RAG 直接对整库做向量检索，在产品手册这种**同一型号不同机型内容高度相似**的场景里，很容易召回混在一起的切片。所以查询图的第一步不是检索，而是：

1. 用 LLM 结合历史对话，抽出用户问题中的产品名并改写问题（`rewritten_query_and_itemnames.prompt`）
2. 把产品名向量化，在 `kb_item_names` 集合中做混合检索
3. 按得分分段：`≥0.85` 进确认集合，`≥0.6` 进候选集合
   - 确认集合有结果 → 后续检索带上 `item_name in [...]` 过滤条件
   - 只有候选 → 直接返回「您想咨询哪个商品 A、B？」
   - 都没有 → 要求用户重新输入

这一步把「检索不到」变成了「问清楚」，避免大模型拿着错误型号的手册硬答。

### 2. 混合检索：稠密 + 稀疏

BGE-M3 一次编码同时产出稠密向量与稀疏向量，入库时都写进 `kb_chunks`：

- 稠密向量 `dense_vector`（dim=1024，COSINE）—— 负责语义相似
- 稀疏向量 `sparse_vector`（IP）—— 负责关键词、型号、数字这类精确匹配

检索时用 `AnnSearchRequest` 构造两路请求，交给 `WeightedRanker` 加权融合，并通过 `norm_score=True` 先把两路分数归一化到同一量级，否则量纲差异会让权重完全失效。切片检索用的权重是 `(0.9, 0.1)`，主体名检索是 `(0.8, 0.2)`。

### 3. 多路召回 + 加权 RRF

三个召回节点从 `node_item_name_confirm` 并行扇出，各自产出切片后汇聚到 `node_rrf`：

```
score(chunk) = Σ  weight_source / (60 + rank_source)
```

不同来源给不同权重，同一切片被多路命中时得分累加，天然实现「多路共识优先」。相比直接比相似度分数，RRF 只用排名不用分值，能规避不同检索器分数量纲不一致的问题。

### 4. Rerank 之后的「断崖截断」

`node_rerank` 不固定返回 TopK，而是先精排，再按分数分布动态决定：

- 硬上限 `RERANK_MAX_TOPK = 10`，下限 `RERANK_MIN_TOPK = 1`
- 当相邻切片的分差超过相对阈值 `RERANK_GAP_RATIO = 0.25` 或绝对阈值 `RERANK_GAP_ABS = 0.5`，认定为「断崖」，在此截断

这样相关切片多时不会漏，相关切片少时也不会硬凑数塞噪声进上下文。

### 5. SSE 流式输出与任务状态

- `sse_utils.py` 维护 `session_id → queue.Queue` 的全局映射；查询图在后台线程执行，`node_answer_output` 通过 `push_to_session` 把 `delta` 事件塞进队列，`sse_generator` 在异步侧消费并通过 `run_in_executor` 阻塞取队列，避免卡住事件循环
- 节点进度由 `task_utils.py` 的内存字典维护（`add_running_task` / `add_done_task`），前端轮询 `/status/{task_id}` 获取，无 IO 开销

### 6. 分块策略

`node_document_split` 分三级处理，参数集中在文件头部：

1. **按标题切分**：用 `RecursiveCharacterTextSplitter` 以 `#` 层级做语义切分；无标题的文档按段落补默认标题
2. **二次切分**：超过 `DEFAULT_MAX_CONTENT_LENGTH = 2000` 的 section 继续切，并保证不产生「只剩标题行的孤儿块」
3. **短块合并**：同一父标题下小于 `MIN_CONTENT_LENGTH = 500` 的碎片块合并回上一块

---

## 测试

```bash
pytest
```

`pytest.ini` 只收集 `tests/` 目录。`app/test/` 下的两个文件是**调试脚本而非测试用例**——它们一 import 就会真实跑完整的导入图 / 查询图（会连 Milvus、MinerU 并消耗 API 额度），所以不在收集范围内。

---

## 已知限制

- 查询图只在 `node_item_name_confirm` 里做主体过滤，`state.py` 中的 `kg_chunks` 字段与 `neo4j_utils.py` 是为图检索预留的，尚未接入主流程
- 任务状态存于进程内存，多进程部署时 `/status` 轮询会失效，需要换成 Redis 等外部存储
- MinerU 走的是云端 API，PDF 解析依赖网络与配额
- `.env` 中的 `MILVUS_METRIC_TYPE`、`MILVUS_MIN_COSINE_SCORE`、`EMBEDDING_DIM` 为预留项，当前代码未读取
- 两个服务需要分别启动，未做统一入口

---

## License

本项目仅用于学习与技术交流。
