# 安装与运行

[返回 README](../README.md)

## 架构

```
学生（网页工作台 / 知秋等 MCP 客户端）──▶ FastAPI + SSE ──▶ 查询链路 ──▶ Milvus（向量 + BM25，jieba 分词）
                                          │                     ├──▶ 本地模型服务（bge-m3，GPU）
                                          │                     └──▶ LLM（默认本地 Ollama，可选 OpenAI 兼容 API）
                                          ├──▶ Redis：按用户限流、GPU 闸门、问答缓存、Celery broker
                                          └──▶ MySQL：用户、文档、章节树、trace
课程资料和笔记上传时同步入库；Celery 只用于 v0.7 的大批量文档库构建      检索范围 = 本人加入的课程 + 本人笔记
```

后端针对的是 LLM 应用的真实约束：**单个请求重、GPU 稀缺、多用户、数据一致性**，不是传统意义上的高并发。理由见 [设计文档](design.md) 1.3 节。

## 本地运行

### 网页工作台（不用命令行）

Windows 上配好 `.env`、拉好 `qwen3:8b` 并完成入库后，双击 `scripts\wenyuan.cmd`：它会依次确认 Docker Desktop、`docker compose`、Milvus 就绪、GPU 模型服务（8001）、专用 Ollama（11435，并行 1、8k 上下文：Ollama 把上下文长度平分给并行槽位，并行 2 时每个请求只有 4096 token）和 API（8000），已在运行的部分直接跳过，然后在浏览器打开 `http://127.0.0.1:8000/ui`。启动 Docker 和模型服务前会先检查可用内存，不够就停下并提示（`-Force` 跳过）。`scripts\wenyuan.cmd -Stop` 全部停掉，包括残留的评测和入库进程。

- **账号**：第一次打开时起个名字就会建号，API key 只存在这个浏览器里；可以记住多个账号，在右上角切换。只有在运行文渊的这台电脑上才能这样建号，别的机器需要粘贴管理员给的 key。
- **问答**：每一步（理解问题 → 查找资料 → 生成并校验答案）实时显示；可以连续追问；`[S1]` 链接到原资料的对应位置；可以选择是否检索自己的笔记。
- **课程**：输入老师给的邀请码加入课程，或者开一门新课。老师可以生成邀请码（只显示一次）、上传和删除课程资料。
- **我的笔记**：上传 `.md` 到自己的私有库，列表、删除。
- **学习目录**：选一个库、填一个主题，看相关章节树和摘要，点章节看全文。
- **看板**：只看得到自己的请求：各阶段耗时、排队时间、p50/p95；点一行看每一步花了多久和一句说明，原始字段在“显示技术细节”里。

### 命令行

需要：Docker、Python 3.12+、[uv](https://docs.astral.sh/uv/)；GPU 模型服务需要 NVIDIA 显卡（50 系需要 CUDA 12.8+ 的 PyTorch）。

```bash
cp .env.example .env
docker compose up -d                                   # Milvus（含 etcd、MinIO）、MySQL、Redis，只监听 127.0.0.1
uv sync && uv run pytest

# GPU 模型服务（在宿主机运行）
cd model_server && uv sync && uv run uvicorn server:app --port 8001

# 课程资料：下载 CS50 数据集（CC BY-NC-SA 4.0，放在 data/ 下，不进仓库）并入库
uvx --from huggingface_hub hf download dev-jonathanb/cs50-educational-rag --repo-type dataset --local-dir data/raw/cs50
uv run python -m app.ingest.cs50
uv run python -m app.ingest.cs50 --links-only           # 只补小节链接（锚点、视频时刻），只用 MySQL

# 课程：开课（指定老师）、生成邀请码（只显示一次）；学生也可以在工作台里用邀请码加入
uv run python -m app.api.auth create teacher           # 打印一次 API key
uv run python -m app.courses create cs50 "CS50x 2024" --teacher teacher
uv run python -m app.courses invite cs50 --by teacher

# 个人笔记：导入一个或多个文件夹到某个账号（子目录的文件按路径命名，可排除）
uv run python -m app.api.auth create alice             # 打印一次 API key
uv run python -m app.ingest.private <笔记文件夹>... --user alice --exclude "草稿*"

# 评测集（出题用 .env 里 JUDGE_* 配置的模型）与评测
uv run python -m eval.build_cs50                       # CS50 自带的 97 题
uv run python -m eval.build_course_eval                # 课程生成题 + 无答案题
uv run python -m eval.build_notes_eval --user alice    # 笔记题
uv run python -m eval.run --exp demo --collection chunks_v2 --mode hybrid --dataset data/eval/cs50_gen_v1.jsonl
uv run python -m eval.run --exp demo-notes --collection chunks_v2 --mode hybrid --dataset data/eval/notes_alice_v1.jsonl --owner alice
uv run python -m eval.judge_retrieval experiments/<一轮结果>.json   # 让独立裁判判断前 5 页是否够用
uv run python -m eval.answers --exp demo --n 100 --dataset data/eval/cs50_gen_v1.jsonl   # 答案层：正确率、逐句有据、引用是否指向标准页
uv run python -m eval.refusal --exp demo                         # 拒答：课程没讲的题是否拒答，能答的题是否误拒（只用本地模型）
uv run python -m eval.gate experiments/<新一轮>.json               # 发版前回归门禁：按评测集自动选基线，显著变差则退出码 1
uv run python -m app.maintenance cleanup                         # 列出 v0.7 遗留和测试数据；加 --yes 才真正删除

# 问答（需要 Ollama：ollama pull qwen3:8b；并行数与 GPU 闸门槽位保持一致，都是 1）
OLLAMA_NUM_PARALLEL=1 OLLAMA_CONTEXT_LENGTH=8192 ollama serve
uv run python -m app.query.ask "How do I free memory from malloc?" --library cs50

# HTTP 服务
uv run uvicorn app.api.main:app --port 8000            # 工作台和看板：http://127.0.0.1:8000/ui
curl -N localhost:8000/v1/ask -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
     -d '{"question": "为什么 C 里没初始化的数组会打印出乱码？", "libraries": ["cs50"]}'
uv run python -m eval.loadtest --label demo --users 1 5 10 20   # 压测
```

v0.7 的官方文档语料（DevDocs、FastAPI 和 Redis 官网）和 SO 评测仍可复现：`app.ingest.devdocs`、`app.ingest.crawl`、`app.ingest.build`、`app.workers.submit`、`eval.build_so`，集合名 `chunks_struct_merge`。
