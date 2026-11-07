# 接入 Agent（MCP）

[返回 README](../README.md)

## 配置

`wenyuan-mcp` 是一个 stdio MCP 服务，它会代理到上面的 HTTP API，所以限流、GPU 闸门和私有库隔离都照常生效。在知秋（Koyo）里，把 `examples/koyo-mcp.json` 复制一份，填好路径和 API key，再用 `ZHIQIU_MCP_CONFIG` 指向它。其他 MCP 客户端的配置方式类似：

```json
{"command": "<wenyuan>/.venv/Scripts/wenyuan-mcp.exe", "env": {"WENYUAN_API_KEY": "<key>"}}
```

## 与知秋（Koyo）的关系

[知秋（Koyo）](https://github.com/zjsyxzjsyx/Koyo)是一个本地优先的学习助手 Agent，负责讲解、自测、间隔复习和学习计划。在课堂场景里，它是学生自己的学习 Agent：通过 MCP 从文渊获取课程章节大纲、资料片段和带引用的回答，检索范围是课程资料加上这个学生自己的笔记。

## 工具

| 工具 | 作用 |
| --- | --- |
| `libraries` | 列出我加入的课程 |
| `search` | 带出处的资料片段 |
| `ask` | 简洁答案 + 引用（经过逐句校验） |
| `outline` | 与主题相关的章节树和 section_id |
| `get_section` | 整节原文 |
| `add_note` / `list_notes` | 存入 / 列出私有笔记 |

输出是紧凑的中文纯文本，每段带出处链接，控制在 5500 字符以内；繁忙、限流、没找到都转成模型能看懂的提示，并附上建议的重试时间。
