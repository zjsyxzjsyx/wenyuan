# 文渊 · wenyuan

> A RAG service over official technical documentation and learners' own notes, with a reproducible evaluation loop and MCP tools for the [Koyo](https://github.com/zjsyxzjsyx/Koyo) learning agent.

文渊是一个面向学习场景的 RAG 服务，名字取自清代皇家藏书楼“文渊阁”：

- **公共文档库**：Python、PostgreSQL、Django、Redis、FastAPI、PyTorch 等 18 套官方技术文档；
- **私有资料库**：学习者自己的笔记和讲义，只有本人可见（开发中）；
- 回答**必须带出处**，资料里没有就拒答；
- 每个设计决策都用评测数据验证。

设计与取舍见 [docs/design.md](docs/design.md)。

## 当前状态（v0.2.0）

| 部分 | 状态 |
|---|---|
| 入库：抓取 / DevDocs → 正文提取 → 按标题分节 → 切块 → 向量化 → Milvus | ✅ 18 个库、9943 页、约 4.6 万 chunk |
| Celery 流水线（编排 / CPU / GPU 三个队列，幂等写入，自动重试） | ✅ |
| 检索：向量、BM25、RRF 混合检索、cross-encoder rerank | ✅ |
| 评测：1204 道 Stack Overflow 真实问题，结果可复现，带显著性检验 | ✅ |
| 全链路 trace（span 树写入 MySQL，可跨 Celery 进程传递） | ✅ |
| 查询理解（本地 Qwen3-8B）、small-to-big 上下文、带引用校验的回答（`python -m app.query.ask`） | ✅ |
| 章节摘要 / “学习者会怎么问”索引（E5） | ✅ 已生成；检索上是负结果，默认关闭（见设计文档 11.2） |
| HTTP API、MCP 工具、私有库、排队限流、trace 看板 | ⏳ 计划中 |

## 评测结果

评测题来自 Stack Overflow：选取**采纳答案链接了官方文档**的问题，以链接到的页面作为检索标准答案。题目是真实学习者提的，不在索引里，不存在数据泄漏。

1184 道题（排除人工审核为“不适用”的题），页面级指标：

| 方案 | Recall@5 | Recall@10 | MRR | 检索延迟 p50 |
|---|---|---|---|---|
| 纯向量（按标题分节 + 合并短小节） | 0.483 | 0.587 | 0.391 | 5 ms |
| 向量 + BM25（RRF 融合） | 0.467 | 0.584 | 0.375 | 5 ms |
| 混合 + rerank 前 50 | 0.490 | 0.617 | 0.390 | 437 ms |
| **混合 + LLM 查询理解**（BM25 用 LLM 推断的 API 名，向量用原问题） | **0.531** | **0.660** | **0.444** | 4 ms + LLM 约 0.8 s |
| 上一行 + rerank 前 50 | 0.507 | 0.643 | 0.407 | 438 ms |

配对 bootstrap 95% 置信区间（`python -m eval.compare A.json B.json`）：
- **直接加 BM25 反而变差**（MRR −1.6，显著）：原问题是夹杂代码和报错的长文本，而且很少出现 API 名。
- **查询理解是最大的提升**：本地 Qwen3-8B 推断出问题背后的 API 名（“删光 Redis”→ `FLUSHALL`），只交给 BM25。相对“混合 + rerank”，四个指标都显著提升，例如 Recall@5 +3.7、MRR +5.2。
- **有了查询理解后，rerank 变成负收益**（Recall@5 −2.4、MRR −3.7，显著），所以默认关闭。
- **章节摘要索引没有带来提升**：摘要只覆盖教程页，按 RRF 等权融合会把 API 参考页挤出前 5（Recall@5 0.531 → 0.474）；降权后最多做到无害。所以默认关闭，摘要留给 `outline` 使用。

几个过程中的发现：
- **数据清洗改变了结论**：清洗前，按标题分节在 Redis 上比固定长度切块更差；排除约一半的噪声页（企业版、发布说明、9 种语言重复的示例）并剔除坏标注后，分节方案反而领先。
- **评测本身也要可复现**：fp16 推理会按 batch 补齐长度，同一段文本在不同 batch 里算出的向量不同，曾让同一实验两次跑分相差约 0.01。现在查询向量统一缓存。
- **reranker 偏爱 API 参考页**：参数名密集的自动生成参考页，会压过答案真正引用的教程页；加了查询理解后 FastAPI 的 Recall@5 从 0.40 跌到 0.30。
- **端到端才暴露的问题**：只有一个 H1 的页面，整页就是一个父块，最长 11 万字符，会独占上下文预算，导致错误拒答；小模型用中文回答时只在 JSON 字段里给引用。两处都已修复，见设计文档 11.2。

## 架构

```
知秋 / MCP 客户端 ──▶ 查询链路 ──▶ Milvus（dense + BM25）
                          ├──▶ 本地模型服务（bge-m3 / bge-reranker-v2-m3，GPU）
                          └──▶ LLM（默认本地 Ollama，可选 OpenAI 兼容 API）
Celery：orchestrate 队列（提取、去重）→ cpu 队列（分节、切块、写 MySQL）→ gpu 队列（向量化、写 Milvus）
MySQL：文档、章节树、trace span    Redis：Celery broker（之后加排队限流、缓存）
```

后端部分针对的是 LLM 应用的真实约束：**单个请求重、GPU 稀缺、多用户、数据一致性**，不是传统意义上的高并发。理由见 [docs/design.md](docs/design.md) 1.3 节。

## 本地运行

需要：Docker、Python 3.12+、[uv](https://docs.astral.sh/uv/)；GPU 模型服务需要 NVIDIA 显卡（50 系需要 CUDA 12.8+ 的 PyTorch）。

```bash
cp .env.example .env
docker compose up -d                                   # Milvus（含 etcd、MinIO）、MySQL、Redis
uv sync && uv run pytest

# GPU 模型服务（在宿主机运行）
cd model_server && uv sync && uv run uvicorn server:app --port 8001

# 准备语料
uv run python -m app.ingest.devdocs                    # 下载 DevDocs 文档集到 data/raw/devdocs/
uv run python -m app.ingest.crawl fastapi              # 抓取官网（FastAPI、Redis）

# 入库：小规模可以同步跑，全量建议用 Celery
uv run python -m app.ingest.build flask --variant struct_merge
uv run celery -A app.workers.celery_app worker -Q orchestrate,cpu --pool=solo -n cpu1@%h   # 可开多个
uv run celery -A app.workers.celery_app worker -Q gpu --pool=solo -n gpu@%h
uv run python -m app.workers.submit python django redis --variant struct_merge --rebuild

# 评测
uv run python -m eval.build_so --tags python django --pages 3 --merge
uv run python -m eval.run --exp demo --collection chunks_struct_merge --mode hybrid --understand --dense-query raw

# 问答（需要 Ollama：ollama pull qwen3:8b）
uv run python -m app.query.ask "How do I delete everything in Redis?" --library redis
```

## 与知秋（Koyo）的关系

[知秋（Koyo）](https://github.com/zjsyxzjsyx/Koyo)是一个本地优先的学习助手 Agent，负责讲解、自测、间隔复习和学习计划。文渊是它的资料后端：知秋通过 MCP 从文渊获取章节大纲、资料片段和带引用的回答。

## 数据与致谢

- 部分代码改编自 [LightRAG](https://github.com/HKUDS/LightRAG)（MIT License，commit `aaebfe1`），协议原文见 `third_party/LICENSES/LightRAG-MIT.txt`；每个改编文件的开头注明了来源和修改内容。
- 公共文档库取自 [DevDocs](https://devdocs.io)，以及 FastAPI、Redis 官网。文档内容的版权归各项目所有，遵循各自的协议（PSF、PostgreSQL License、BSD、Apache 2.0、CC BY-SA 等）。本仓库**不分发**文档原文，使用者运行脚本自行下载。
- 评测集 `eval/datasets/so_v1.jsonl` 来自 [Stack Overflow](https://stackoverflow.com)，遵循 CC BY-SA 4.0；每行都保留了原问题和答案的链接。

## License

代码采用 [MIT](LICENSE) 协议。
