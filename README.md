# 文渊

中文 | [English](README.en.md)

文渊是一门课共用的学习资料 RAG 服务，名字取自清代皇家藏书楼“文渊阁”。老师上传的课程资料全班共享，每个学生的笔记只有自己能检索到；提问时同时检索两者，回答逐句标明出处，资料里没有就拒答。系统部署在一台带 GPU 的服务器上全班共用，学生通过网页工作台或自己的学习 Agent（如[知秋](https://github.com/zjsyxzjsyx/Koyo)，经 MCP）使用。演示课程是哈佛 CS50x 2024。

## 功能

- **课程与角色**：任何人都能开课成为老师；老师上传资料、发邀请码（只存哈希），学生凭邀请码加入。检索范围以学生加入的课程为上限，没加入的课程一律 403。
- **个人笔记**：上传或按文件夹导入 Markdown，只有本人能检索到；只要用到了笔记，就只用本地模型回答。
- **带出处的回答**：每句标 `[S#]`，讲义跳到段落锚点，讲课转写跳到视频的具体时刻；回答逐句和资料核对，无依据的句子删掉，课程没讲的问题拒答。
- **检索**：Milvus 混合检索（向量 + BM25，jieba 中文分词），本地模型先做查询理解，推断问题背后的术语和 API 名；多跳问题按需拆分。
- **一块 GPU 全班共用**：Redis 实现的按用户限流、公平排队的 GPU 闸门与准入控制、问答缓存与防击穿。
- **网页工作台**：问答（阶段实时显示）、我的笔记、学习目录、课程管理、老师端全班高频问题汇总、只看自己请求的看板。
- **MCP**：7 个工具，供知秋等学习 Agent 调用，限流和权限照常生效。
- **评测**：按“证据原文”计分的检索评测、答案层评测（正确率、逐句有据、有害内容）、拒答测试、发版门禁。

## 效果

100 道 CS50 课程题，本地 Qwen3-8B 回答，DeepSeek 当评委：

| 指标 | 结果 |
|---|---|
| 正确证据在前 5 个检索片段里 | 86.6%（混合检索 74.8%，加查询理解后） |
| 回答完全正确 | 86% |
| 带有害内容（与资料矛盾或错误）的回答 | 4% |
| 课程没讲的问题拒答 / 能答的问题误拒 | 100% / 2.5% |
| 其他学生检索到你的笔记 | 0 条 |

每项结果的实验过程见[评测与结果](docs/evaluation.md)。

## 快速开始

需要 Docker、Python 3.12+、[uv](https://docs.astral.sh/uv/)、[Ollama](https://ollama.com) 和一块 NVIDIA 显卡（50 系需要 CUDA 12.8+ 的 PyTorch）。

```powershell
copy .env.example .env                     # 按注释填写
uv sync
ollama pull qwen3:8b
uvx --from huggingface_hub hf download dev-jonathanb/cs50-educational-rag --repo-type dataset --local-dir data/raw/cs50
scripts\wenyuan.cmd                        # 依次启动 Docker、模型服务、Ollama 和 API，打开 http://127.0.0.1:8000/ui
uv run python -m app.ingest.cs50           # 第一次：CS50 资料入库
uv run python -m app.api.auth create teacher                       # 老师账号（打印一次 API key）
uv run python -m app.courses create cs50 "CS50x 2024" --teacher teacher
```

第一次打开工作台时起个名字就能建号；学生凭老师生成的邀请码在「课程」页加入。`scripts\wenyuan.cmd -Stop` 全部停掉。启动前会检查可用内存，不够就停下提示。

## 常用命令

```powershell
uv run python -m app.api.auth create <账号>                                # 建号（打印一次 API key）
uv run python -m app.courses invite cs50 --by <老师账号>                    # 生成邀请码（只显示一次）
uv run python -m app.courses enroll cs50 <账号>                            # 直接把账号加入课程
uv run python -m app.ingest.private <笔记文件夹> --user <账号>               # 导入个人笔记
uv run python -m app.query.ask "How do I free memory from malloc?" --library cs50
uv run python -m eval.run --exp demo --collection chunks_v2 --mode hybrid --dataset data/eval/cs50_gen_v1.jsonl   # 检索评测
uv run python -m eval.answers --exp demo --dataset data/eval/cs50_gen_v1.jsonl   # 答案层评测
uv run python -m eval.refusal --exp demo                                   # 拒答测试
uv run python -m eval.gate experiments/<新一轮>.json                        # 发版门禁
uv run python -m app.maintenance cleanup                                   # 列出测试数据；加 --yes 才删除
```

完整的命令和每一步的说明见[安装与运行](docs/running.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [设计文档](docs/design.md) | 方向与复盘、场景、架构、入库与查询链路、权限、评测方法；每个里程碑的实验记录 |
| [评测与结果](docs/evaluation.md) | 评测集、检索 / 答案 / 拒答结果、v0.7 的实验、压测 |
| [安装与运行](docs/running.md) | 架构图、网页工作台、全部命令、v0.7 语料的复现 |
| [接入 Agent（MCP）](docs/mcp.md) | MCP 配置、7 个工具、与知秋的关系 |

## 数据与隐私

- 学生的笔记只有本人能检索到；检索、缓存键和 trace 都按用户隔离，上下文里有笔记时只用本地模型。
- API key 只存 SHA-256；邀请码只存哈希；每个人在看板里只看得到自己的请求。
- 老师看到的高频问题只显示至少两位学生问过的类别，不显示是谁问的。
- 外部模型（DeepSeek）只在评测时用来出题和评分，只发送课程的公开资料；笔记只在作者同意时用于出题。
- 课程原文、评测题和回答原文都只在本地（`data/`、`.cache/`），不进仓库；`experiments/` 里只有题号、得分和汇总。

## 开发

```powershell
uv sync
uv run pytest -q              # 单元测试不需要模型和服务
uv run python -m eval.gate experiments/<新一轮>.json
```

完整评测需要 GPU、语料和本地模型，所以托管 CI 只跑单元测试，发版前在本机跑评测门禁。

## 致谢与许可

- 部分代码改编自 [LightRAG](https://github.com/HKUDS/LightRAG)（MIT License，commit `aaebfe1`），协议原文见 `third_party/LICENSES/LightRAG-MIT.txt`；每个改编文件的开头注明了来源和修改内容。
- 演示课程资料来自哈佛 [CS50x](https://cs50.harvard.edu/x/)，经 Hugging Face 数据集 [dev-jonathanb/cs50-educational-rag](https://huggingface.co/datasets/dev-jonathanb/cs50-educational-rag) 整理成 Markdown，遵循 CC BY-NC-SA 4.0。本仓库**不分发**课程原文和由它生成的评测题，使用者运行命令自行下载。
- v0.7 的公共文档取自 [DevDocs](https://devdocs.io)，以及 FastAPI、Redis 官网，版权归各项目所有，同样不在仓库中分发；评测集 `eval/datasets/so_v1.jsonl` 来自 [Stack Overflow](https://stackoverflow.com)，遵循 CC BY-SA 4.0，每行都保留了原问题和答案的链接。
- 代码采用 [MIT](LICENSE) 协议。
