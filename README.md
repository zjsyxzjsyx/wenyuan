# 文渊 · wenyuan

> A RAG service for a classroom: the course's shared materials plus each student's private notes, answers with citations, one shared GPU, and MCP tools for learning agents such as [Koyo](https://github.com/zjsyxzjsyx/Koyo).

文渊是一门课共用的学习资料 RAG 服务，名字取自清代皇家藏书楼“文渊阁”：

- **课程资料**：老师上传，全班共享。演示用的是哈佛 CS50x 2024 的讲义、讲课转写和作业；
- **个人笔记**：每个学生上传自己的笔记，只有本人能检索到；
- 问答**同时检索两者，并标明出处**，资料里没有就拒答；
- 部署在一台带 GPU 的服务器上全班共用，学生通过网页工作台或自己的学习 Agent（经 MCP）使用；
- 每个设计决策都用评测数据验证。

**为什么不只靠联网搜索**：课程讲义网上搜不到，答案要和老师讲的、考的保持一致；学生自己的笔记更不可能在网上。**为什么要多用户、排队和缓存**：几十个学生共用一块 GPU，作业截止前会集中提问；全班同一周学同一章，问题重复度高。

设计与取舍见 [docs/design.md](docs/design.md)（第 0 节是 v2 的方向与复盘）。

## 当前状态（v2 进行中）

项目一开始做的是“官方技术文档 + 个人笔记”（v0.7，`git tag v0.7.0`）。做下来发现：只有公开文档时，联网搜索往往就够用；真正不可替代的是课程自己的资料和学生的笔记。所以 v2 改成课堂场景，原有的后端架构保留，数据和评测重新做。

| 部分 | 状态 |
|---|---|
| CS50 课程资料入库：70 个文件、1435 个片段；引用指向 CS50 原页面或讲课视频 | ✅ `python -m app.ingest.cs50` |
| 个人笔记：文件夹递归导入；按用户隔离检索、缓存和 trace；有私有内容时只用本地模型生成答案 | ✅ 其他用户检索到私有内容 0 条 |
| 中英混合检索：BM25 改用 jieba 中文分词（原分词器把一整句中文当成一个词） | ✅ |
| 评测：课程生成题 238 道、笔记题 160 道、无答案题 40 道，按“证据原文”计分 | ✅ 见下文 |
| 混合检索、查询理解、HyDE、LLM 重排、论断级引用校验、多跳检索 | ✅ 已实现；HyDE 和 LLM 重排在新评测集上验证中 |
| Redis：按用户限流、GPU 闸门（公平排队 + 准入控制）、问答缓存 + 防击穿 | ✅ v0.7 压测验证，待用课程数据重测 |
| 网页工作台（问答、我的笔记、学习目录、只看自己请求的 trace 看板）、MCP 服务（7 个工具） | ✅ |
| 课程与角色：任何人可以开课并成为老师；老师上传资料、发邀请码（只存哈希），学生凭邀请码加入；检索只在自己的课程内，访问没加入的课程一律 403 | ✅ 工作台「课程」页、`python -m app.courses` |
| 老师端：全班高频问题汇总（只看汇总，不看个人笔记） | ⏳ |

## 评测（v2）

评测题由 **DeepSeek** 出，和检索链路里的本地模型不是同一个，避免“自己出题自己答”。题目都放在 `data/eval/`，不进仓库（课程资料的许可证不允许，笔记是个人内容）。

| 评测集 | 题数 | 怎么来的 |
|---|---|---|
| CS50 自带题 | 97 | 原数据集自带：模型出题、人工打分，标注了出处文件和原文片段 |
| 课程生成题 | 238 | 按课程段落出题：**换说法**（描述想做什么、遇到什么问题，不说术语）98 道、**中文提问** 80 道、**跨周综合** 60 道 |
| 课程无答案题 | 40 | 课程没讲的内容，用于拒答测试 |
| 笔记题 | 160 | 按一位学生的笔记出题：回忆 60、概念 60、笔记 + 课程混合 40；检索范围是“课程 + 自己的笔记” |

**按证据计分**：检索到的片段要和出题所用的原文有足够重合（连续 5 个词或 5 个汉字，重合比例 ≥ 0.5）才算命中，和切块方式无关。课程只有 70 个页面，“找对页面”的比例都在 0.9 以上，分不出好坏。

| 前 5 个片段里有证据 | 向量 | BM25 | 混合 |
|---|---|---|---|
| CS50 自带题（有原文片段的 66 道） | 0.894 | 0.909 | **0.939** |
| 课程生成题 | **0.794** | 0.567 | 0.748 |
| 　其中中文提问 | | | 0.675 |
| 笔记题 | 0.775 | **0.869** | 0.844 |
| 　其中混合题（两段证据都在前 10） | | | 0.525 |

- **自带题太容易**：题目照着原文写，三种方式都在 0.9 左右，只能作参照。
- **没有一种检索方式到处最好**：英文课程配上换说法和中文的提问，几乎没有共同的词，向量检索最好；中文问中文笔记，jieba 分词后 BM25 最好。
- **最弱的是中文问英文课程和需要两段证据的题**，接下来的查询理解、HyDE、LLM 重排先在这几类题上验证。目标：课程题和笔记题都 ≥ 85%。

## v0.7 实验（官方文档 + Stack Overflow）

v0.7 的语料是 18 套官方技术文档（约 4.6 万个片段），评测题是 1184 道 Stack Overflow 真实问题：选取**采纳答案链接了官方文档**的问题，以链接到的页面作为标准答案。这些实验的结论仍然适用于现在的检索链路。所有比较都做了配对 bootstrap 95% 置信区间（`python -m eval.compare A.json B.json`）。

### 检索

| 方案 | Recall@5 | MRR |
|---|---|---|
| 纯向量（按标题分节 + 合并短小节） | 0.483 | 0.391 |
| 向量 + BM25（RRF 融合） | 0.467 | 0.375 |
| 混合 + cross-encoder 重排前 50 | 0.490 | 0.390 |
| **混合 + LLM 查询理解**（BM25 用 LLM 推断的 API 名，向量用原问题） | **0.531** | **0.444** |
| 混合 + 查询理解 + cross-encoder 重排 | 0.507 | 0.407 |
| 混合 + 查询理解 + **LLM 重排前 40 页** | 0.612 | 0.542 |
| 混合 + 查询理解 + **HyDE**（假想答案同时加进向量和 BM25 查询） | **0.624** | 0.517 |

- **直接加 BM25 反而变差**：原问题是夹杂代码和报错的长文本，很少出现 API 名。**查询理解是最大的单项提升**：本地 Qwen3-8B 推断出问题背后的 API 名（“删光 Redis”→ `FLUSHALL`），只交给 BM25。
- **cross-encoder 重排是负收益**：它只看片段和问题像不像，偏爱参数名密集的 API 参考页。**换成 LLM 重排后反而显著提升**：大模型一次看完所有候选页，按“能不能写出正确答案”来排。
- **HyDE**：先让模型写一段假想答案再去检索，把问题里没提到的功能名（比如“后台跑线程”→ lifespan events）带进检索，Recall@5 +9.3（显著）。
- **SO 的标准答案太窄**：只认答主链接的那一页。让 DeepSeek 判断“前 5 页里有没有一页够用”，基线是 77%，LLM 重排后 87.5%。这也是 v2 改成按证据计分的原因。
- **章节摘要索引没有带来提升**，**数据清洗改变了切块实验的结论**，**评测本身也要可复现**（查询向量统一缓存），详见设计文档 11.2。

### 答案、多轮、多跳

| 实验 | 结果 |
|---|---|
| **论断级引用校验**（逐句核对引用的原文，不支持就删掉）：100 道题，DeepSeek 当独立裁判 | 含无依据论断的答案 47.8% → **23.0%**（显著），完全正确率不变（44%），多花约 0.85 s |
| **多轮对话**：199 道两轮题，追问里把对象换成「它 / 那个」 | 只有追问 Recall@5 0.215；带上对话历史 **0.488**，收回 83% 的损失 |
| **跨库多跳**：66 道需要同时查两个库的问题 | 单次检索两页都在前 10：0.303；拆成子查询、按页面轮流交错：**0.455**（显著）。对所有问题都拆会伤害单库问题，被发版门禁拦下 |

### 换向量模型：一个评测集会骗人

把 BGE-m3 换成 Qwen3-Embedding-0.6B：在 SO 题上 Recall@5 **+11.8**（显著），在 DeepSeek 新写的两个评测集上一个不显著、一个显著**更差**（−6.8）。SO 上的领先很可能来自训练数据见过 Stack Overflow，所以默认仍用 BGE-m3。

### 共享部署：一块 GPU，20 人同时提问

`python -m eval.loadtest`：20 个用户，每人连续问 3 个问题，客户端等待 30 秒后放弃。

| | 不加闸门 | GPU 闸门（2 槽）+ 准入控制 |
|---|---|---|
| 成功 / 超时 | 13 / **47** | 19 / **0** |
| 有效吞吐（30 s 内完成的答案） | 8.7 个/分钟 | **21.9 个/分钟** |
| 首个事件 p95 | 27.9 s | 0.21 s |
| GPU 显存不足（OOM） | 2 次，Ollama 崩溃 | 0 |

不加闸门时会出现拥塞崩溃：客户端已经放弃的请求还在占用 GPU。加上闸门后，超出能力的请求在 0.1 秒内得到 `busy` 和建议的重试时间。压测还抓出 4 个单用户测不出的 bug，见设计文档 11.3。

## 架构

```
学生（网页工作台 / 知秋等 MCP 客户端）──▶ FastAPI + SSE ──▶ 查询链路 ──▶ Milvus（向量 + BM25，jieba 分词）
                                          │                     ├──▶ 本地模型服务（bge-m3，GPU）
                                          │                     └──▶ LLM（默认本地 Ollama，可选 OpenAI 兼容 API）
                                          ├──▶ Redis：按用户限流、GPU 闸门、问答缓存、Celery broker
                                          └──▶ MySQL：用户、文档、章节树、trace
Celery：课程资料入库（提取、切块、向量化）       检索范围 = 课程资料 + 本人笔记（按 owner_id 过滤）
```

后端针对的是 LLM 应用的真实约束：**单个请求重、GPU 稀缺、多用户、数据一致性**，不是传统意义上的高并发。理由见 [docs/design.md](docs/design.md) 1.3 节。

## 本地运行

### 网页工作台（不用命令行）

Windows 上配好 `.env`、拉好 `qwen3:8b` 并完成入库后，双击 `scripts\wenyuan.cmd`：它会依次确认 Docker Desktop、`docker compose`、Milvus 就绪、GPU 模型服务（8001）、专用 Ollama（11435，并行 2、8k 上下文）和 API（8000），已在运行的部分直接跳过，然后在浏览器打开 `http://127.0.0.1:8000/ui`。`scripts\wenyuan.cmd -Stop` 全部停掉。

- **账号**：第一次打开时起个名字就会建号，API key 只存在这个浏览器里；可以记住多个账号，在右上角切换。只有在运行文渊的这台电脑上才能这样建号，别的机器需要粘贴管理员给的 key。
- **问答**：每一步（理解问题 → 查找资料 → 生成并校验答案）实时显示；可以连续追问；`[S1]` 链接到原资料的对应位置；可以选择是否检索自己的笔记。
- **课程**：输入老师给的邀请码加入课程，或者开一门新课。老师可以生成邀请码（只显示一次）、上传和删除课程资料。
- **我的笔记**：上传 `.md` 到自己的私有库，列表、删除。
- **学习目录**：选一个库、填一个主题，看相关章节树和摘要，点章节看全文。
- **看板**：只看得到自己的请求：各阶段耗时、排队时间、p50/p95，点一行看 span 瀑布图。

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
uv run python -m eval.gate experiments/<新一轮>.json               # 发版前回归门禁：显著变差则退出码 1

# 问答（需要 Ollama：ollama pull qwen3:8b；并行槽位与 GPU 闸门保持一致）
OLLAMA_NUM_PARALLEL=2 OLLAMA_CONTEXT_LENGTH=8192 ollama serve
uv run python -m app.query.ask "How do I free memory from malloc?" --library cs50

# HTTP 服务
uv run uvicorn app.api.main:app --port 8000            # 工作台和看板：http://127.0.0.1:8000/ui
curl -N localhost:8000/v1/ask -H "Authorization: Bearer <key>" -H "Content-Type: application/json" \
     -d '{"question": "为什么 C 里没初始化的数组会打印出乱码？", "libraries": ["cs50"]}'
uv run python -m eval.loadtest --label demo --users 1 5 10 20   # 压测
```

v0.7 的官方文档语料（DevDocs、FastAPI 和 Redis 官网）和 SO 评测仍可复现：`app.ingest.devdocs`、`app.ingest.crawl`、`app.ingest.build`、`app.workers.submit`、`eval.build_so`，集合名 `chunks_struct_merge`。

### 接入 Agent（MCP）

`wenyuan-mcp` 是一个 stdio MCP 服务，它会代理到上面的 HTTP API，所以限流、GPU 闸门和私有库隔离都照常生效。在知秋（Koyo）里，把 `examples/koyo-mcp.json` 复制一份，填好路径和 API key，再用 `ZHIQIU_MCP_CONFIG` 指向它。其他 MCP 客户端的配置方式类似：

```json
{"command": "<wenyuan>/.venv/Scripts/wenyuan-mcp.exe", "env": {"WENYUAN_API_KEY": "<key>"}}
```

## 与知秋（Koyo）的关系

[知秋（Koyo）](https://github.com/zjsyxzjsyx/Koyo)是一个本地优先的学习助手 Agent，负责讲解、自测、间隔复习和学习计划。在课堂场景里，它是学生自己的学习 Agent：通过 MCP 从文渊获取课程章节大纲、资料片段和带引用的回答，检索范围是课程资料加上这个学生自己的笔记。

## 数据与致谢

- 部分代码改编自 [LightRAG](https://github.com/HKUDS/LightRAG)（MIT License，commit `aaebfe1`），协议原文见 `third_party/LICENSES/LightRAG-MIT.txt`；每个改编文件的开头注明了来源和修改内容。
- 演示课程资料来自哈佛 [CS50x](https://cs50.harvard.edu/x/)，经 Hugging Face 数据集 [dev-jonathanb/cs50-educational-rag](https://huggingface.co/datasets/dev-jonathanb/cs50-educational-rag) 整理成 Markdown，遵循 CC BY-NC-SA 4.0。本仓库**不分发**课程原文和由它生成的评测题，使用者运行命令自行下载。
- v0.7 的公共文档取自 [DevDocs](https://devdocs.io)，以及 FastAPI、Redis 官网，版权归各项目所有，同样不在仓库中分发；评测集 `eval/datasets/so_v1.jsonl` 来自 [Stack Overflow](https://stackoverflow.com)，遵循 CC BY-SA 4.0，每行都保留了原问题和答案的链接。

## License

代码采用 [MIT](LICENSE) 协议。
