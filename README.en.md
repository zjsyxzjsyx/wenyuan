# Wenyuan (文渊)

[中文](README.md) | English

Wenyuan is a retrieval-augmented question-answering service for one class. The teacher's course material is shared by the whole class, and each student's notes are searchable by that student only. A question searches both, every sentence of the answer cites where it came from, and a question the material does not cover is declined. One GPU server serves the class; students use the web workbench or their own learning agent (such as [Koyo](https://github.com/zjsyxzjsyx/Koyo)) over MCP. The demo course is Harvard's CS50x 2024. The name comes from Wenyuan Pavilion, the Qing dynasty's imperial library.

> The workbench, the prompts and the detailed documentation are in **Chinese**; the course material and most evaluation questions are in English.

## Design

- **The scope is a ceiling.** The API passes the caller's courses into the query path. The model may narrow a search to some of them, never widen it. A course the caller has not joined is a 403, and a section of another course is a 404, as if it did not exist.
- **Notes stay private and local.** Retrieval, cache keys and traces are separated per user. When a student's notes are in the context, only the local model sees them.
- **Every sentence is checked.** The answer is split into sentences, and one model call checks each one against all retrieved sources. Unsupported sentences are removed, not rewritten. An answer that loses most of its sentences becomes a refusal.
- **Find the bottleneck before adding parts.** On 100 course questions, the evidence was already in the top five passages for 88, yet only 60 were answered correctly: the losses were in generation, not retrieval. Attributing them one by one found engineering defects, not model limits. The checker saw only the 2 cited sources out of 11. Course questions were taken for small talk and never searched. JSON output was cut at an unescaped quote in code. Fixing them raised fully correct answers from 61.6% to 86%.
- **A scarce GPU, not high concurrency.** Requests are few and heavy. A fair-queueing GPU gate in Redis (Lua) with admission control answers "busy" with a retry time instead of letting requests time out. Per-user rate limits and an answer cache with stampede protection sit in front.
- **Measured, not assumed.** Retrieval is scored on the evidence passage, not the page. Every comparison is a paired bootstrap on the same questions. A release gate fails on any significant regression. Reranking and HyDE stay off because they did not help here.

## Results

100 CS50 course questions, answered by a local Qwen3-8B, judged by DeepSeek:

| Measure | Result |
|---|---|
| Evidence in the top 5 retrieved passages | 86.6% (74.8% with hybrid search alone) |
| Fully correct answers | 86% |
| Answers with a harmful sentence (contradicts the material, or wrong) | 4% |
| Out-of-course questions declined / answerable questions declined | 100% / 2.5% |
| Another student's notes retrieved | 0 |

## Features

- Courses and roles: anyone can open a course; teachers upload material and issue invite codes (stored hashed); students join with a code.
- Private notes: upload or import folders of Markdown.
- Citations that land on the paragraph anchor of the lecture notes or the second of the lecture video.
- Hybrid search in Milvus (dense + BM25 with jieba tokenization), query understanding by a local model, multi-hop decomposition on demand.
- Web workbench (FastAPI and SSE): questions with live stages, notes, course outline, course management, the teacher's summary of frequent class questions, and a dashboard of the caller's own requests.
- MCP server with 7 tools.

## Quick start

Requires Docker, Python 3.12+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com) and an NVIDIA GPU (RTX 50-series cards need PyTorch built for CUDA 12.8+).

```powershell
copy .env.example .env                     # fill in as commented
uv sync
ollama pull qwen3:8b
uvx --from huggingface_hub hf download dev-jonathanb/cs50-educational-rag --repo-type dataset --local-dir data/raw/cs50
scripts\wenyuan.cmd                        # starts Docker, the model server, Ollama and the API; opens http://127.0.0.1:8000/ui
uv run python -m app.ingest.cs50           # first time: index CS50
uv run python -m app.api.auth create teacher
uv run python -m app.courses create cs50 "CS50x 2024" --teacher teacher
```

`scripts\wenyuan.cmd -Stop` stops everything.

Common commands:

```powershell
uv run python -m app.courses invite cs50 --by teacher                    # an invite code, shown once
uv run python -m app.ingest.private <notes folder> --user <name>         # import a student's notes
uv run python -m app.query.ask "How do I free memory from malloc?" --library cs50
uv run python -m eval.answers --exp demo --dataset data/eval/cs50_gen_v1.jsonl
uv run python -m eval.refusal --exp demo
uv run python -m eval.gate experiments/<new run>.json
```

## Data and privacy

- Course texts, evaluation questions and answer texts stay on the local machine (`data/`, `.cache/`). The repository's `experiments/` keeps question ids, scores and summaries only.
- API keys and invite codes are stored as hashes. Each user sees only their own requests in the dashboard. The teacher's summary shows a question only when at least two students asked it, and never who asked.
- An external model (DeepSeek) is used only for evaluation, to write and grade questions, and only on the public course material. Notes were used for questions only with their author's consent.

## Development

```powershell
uv sync
uv run pytest -q              # unit tests need no model or service
```

Full evaluations need the GPU, the corpus and the local models, so hosted CI runs the unit tests and the release gate is run locally before a release.

The detailed documentation is in Chinese under [`docs/`](docs/): [design](docs/design.md), [evaluation and results](docs/evaluation.md), [installation and commands](docs/running.md) and [MCP](docs/mcp.md).

## Credits and license

- Parts of the code are adapted from [LightRAG](https://github.com/HKUDS/LightRAG) (MIT License, commit `aaebfe1`); see `third_party/LICENSES/LightRAG-MIT.txt`. Each adapted file names its source and changes at the top.
- Course material: Harvard's [CS50x](https://cs50.harvard.edu/x/), as Markdown in the Hugging Face dataset [dev-jonathanb/cs50-educational-rag](https://huggingface.co/datasets/dev-jonathanb/cs50-educational-rag), CC BY-NC-SA 4.0. This repository does **not** distribute the course text or the questions generated from it.
- v0.7's public documentation came from [DevDocs](https://devdocs.io) and the FastAPI and Redis sites, and is not redistributed. Its evaluation set came from [Stack Overflow](https://stackoverflow.com) (CC BY-SA 4.0), every row linking to its original question and answer. v0.7's code and evaluation sets are no longer on the main branch; they are under `git tag v0.7.0`.
- The code is under the [MIT](LICENSE) license.
