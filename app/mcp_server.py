"""wenyuan as an MCP server (stdio), for agents such as Koyo.

`WENYUAN_API_KEY=<key> python -m app.mcp_server`   (WENYUAN_URL defaults to http://127.0.0.1:8000)

A thin proxy over the HTTP API, deliberately: every tool call passes the same
guards as any other client (per-user rate limit, GPU gate, answer cache, private
notes visible only to the key's owner), so MCP is not a side door around them.

Tools return compact plain text with a source URL for every passage: agents put
tool output into the model's context as one block of text (Koyo caps it at 6000
characters by default), so the formatting here decides what the model can cite.
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

BUDGET = 5500     # characters per tool result, under Koyo's default 6000 cap

mcp = FastMCP("wenyuan", instructions=(
    "文渊：官方技术文档（Python、Redis、PostgreSQL、Django、FastAPI、Docker、PyTorch 等 18 个库）"
    "和用户自己笔记的检索服务。讲课、出题前用 search 取带出处的原文；学习者提问用 ask 拿带引用的简短答案；"
    "做学习计划用 outline 看章节结构，再用 get_section 取整节内容。资料里没有的内容不要编造。"))


class WenyuanError(RuntimeError):
    pass


def _client() -> httpx.Client:
    key = os.environ.get("WENYUAN_API_KEY", "")
    if not key:
        raise WenyuanError("WENYUAN_API_KEY is not set (create one with `python -m app.api.auth create <name>`)")
    return httpx.Client(base_url=os.environ.get("WENYUAN_URL", "http://127.0.0.1:8000"),
                        headers={"Authorization": f"Bearer {key}"}, timeout=120)


def _call(method: str, path: str, **kwargs: Any) -> Any:
    with _client() as client:
        try:
            response = client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise WenyuanError(f"文渊服务不可用：{type(exc).__name__}") from exc
    if response.status_code == 200:
        return response.json()
    detail = response.json().get("detail") if response.headers.get("content-type", "").startswith("application/json") else None
    kind = detail.get("type") if isinstance(detail, dict) else None
    retry = response.headers.get("Retry-After")
    if kind == "busy" or response.status_code == 429:
        raise WenyuanError(f"文渊正忙（{kind or 'rate_limited'}），请在 {retry or '几'} 秒后重试")
    if response.status_code == 404:
        raise WenyuanError("没有找到（不存在，或者不是你的私有笔记）")
    raise WenyuanError(f"文渊返回 {response.status_code}：{detail}")


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …（已截断）"


def _turns(history: list[dict] | None) -> list[dict]:
    return [{"role": t.get("role"), "content": str(t.get("content", ""))} for t in (history or [])
            if t.get("role") in ("user", "assistant")][-6:]


def format_sources(sources: list[dict], budget: int = BUDGET) -> str:
    if not sources:
        return "资料里没有找到相关内容。"
    per = max(300, budget // len(sources) - 120)
    blocks = []
    for s in sources:
        origin = "（我的笔记）" if s.get("private") else ""
        head = f"[{s['sid']}] {s['title']} > {s['heading_path']}{origin}\n{s['url']}"
        blocks.append(f"{head}\n{_clip(s.get('text', ''), per)}")
    return _clip("\n\n".join(blocks), budget)


def format_answer(payload: dict) -> str:
    answer = payload["answer"]
    lines = [answer["text"]]
    if answer["citations"]:
        lines.append("\n出处：")
        lines += [f"[{c['sid']}] {c['title']} > {c['heading_path']} — {c['url']}" for c in answer["citations"]]
    elif not answer["found"]:
        lines.append("\n（资料里没有可靠依据，不要凭记忆补充。）")
    return "\n".join(lines)


def format_outline(docs: list[dict], budget: int = BUDGET) -> str:
    if not docs:
        return "没有找到相关章节。"
    lines = []
    for d in docs:
        lines.append(f"《{d['title']}》 {d['url']}")
        for s in d["sections"]:
            lines.append(f"{'  ' * max(s['level'] - 1, 0)}- {s['heading']}  [{s['section_id']}]")
            if s.get("summary"):
                lines.append(f"{'  ' * s['level']}摘要：{_clip(s['summary'], 200)}")
    return _clip("\n".join(lines), budget)


@mcp.tool()
def libraries() -> str:
    """列出用户所在的课程 id（search/ask/outline 的 libraries 参数用这些 id；"private" 是用户自己的笔记）。"""
    rows = [f"{c['id']}: {c['name']}（{'老师' if c.get('role') == 'teacher' else '学生'}）" for c in _call("GET", "/v1/courses")]
    return "\n".join(rows or ["（还没有加入任何课程）"]) + "\nprivate: 我的笔记"


@mcp.tool()
def search(question: str, libraries: list[str] | None = None, history: list[dict] | None = None) -> str:
    """检索带出处的原文片段（不生成答案），用于讲解、出题。libraries 可选，如 ["cs50"]，默认是用户所在的全部课程；history 是最近几轮对话
    [{"role": "user"|"assistant", "content": ...}]，用于理解"它""那个"指什么。"""
    body = {"question": question, "libraries": libraries or None, "history": _turns(history)}
    return format_sources(_call("POST", "/v1/search", json=body)["sources"])


@mcp.tool()
def ask(question: str, libraries: list[str] | None = None, history: list[dict] | None = None) -> str:
    """回答学习者的问题：简短答案 + 课程资料或笔记的出处；资料里没有就明确说没有。参数同 search。"""
    body = {"question": question, "libraries": libraries or None, "history": _turns(history), "stream": False}
    return format_answer(_call("POST", "/v1/ask", json=body))


@mcp.tool()
def outline(library: str, topic: str | None = None) -> str:
    """查看一门课程（或 "private" 我的笔记）的章节结构，用于制定学习计划。给 topic（如 "持久化"）时只列相关页面。
    每个小节带 [section_id]，可用 get_section 取整节内容。"""
    return format_outline(_call("GET", "/v1/outline", params={"library": library, **({"topic": topic} if topic else {})}))


@mcp.tool()
def get_section(section_id: str) -> str:
    """取一个小节的完整内容（含其下的子小节），section_id 来自 outline，形如 "1a2b3c...:4"。"""
    s = _call("GET", f"/v1/sections/{section_id}")
    tail = "\n（内容过长，已截断）" if s["truncated"] else ""
    return _clip(f"{s['title']} > {s['heading_path']}\n{s['url']}\n\n{s['text']}", BUDGET) + tail


@mcp.tool()
def add_note(filename: str, content: str) -> str:
    """把用户自己的 Markdown 笔记存入私有库（只有本人能检索到；同名文件会被覆盖）。"""
    r = _call("POST", "/v1/notes", json={"filename": filename, "content": content})
    return f"已保存《{r['title']}》（{r['filename']}，{r['chunks']} 个片段），之后的 search/ask 会一并检索。"


@mcp.tool()
def list_notes() -> str:
    """列出用户已存入私有库的笔记。"""
    notes = _call("GET", "/v1/notes")
    return "\n".join(f"{n['filename']}：{n['title']}（{n['doc_id']}）" for n in notes) or "私有库还没有笔记。"


def main() -> None:
    mcp.run()   # stdio


if __name__ == "__main__":
    main()
