# WozClaw

一个最小可运行、可观察、可验证的记忆型 Agent Demo。

技术栈：
- AgentScope ReAct Agent
- 文件系统 Memory Store（长期 / 会话 / 每日 / 日志）
- FastAPI + 前端单页聊天界面

本项目的重点不是“聊天壳子”，而是 memory 的工程化设计：
- 记忆如何分层
- 如何用 `message_id` 做锚点检索
- 如何窗口扩展上下文
- 工具调用如何持久化并在前端回放

## Memory 设计总览

### 1. 分层模型

按 `user_id` 隔离，核心数据位于 `memory/{user_id}/`：

```text
memory/
  {user_id}/
    memory.md                   # 长期记忆（整份文本）
    conversations.json          # 会话列表索引（title + updated_at）
    sessions/
      {session_id}.jsonl        # 会话短期记忆
    daily/
      YYYY-MM-DD.jsonl          # 当日记忆
    logs/
      llm_dialogue.jsonl        # 每轮完整日志（含工具调用）
```

分层职责：
- 长期记忆（`memory.md`）：稳定偏好、长期事实。
- 会话记忆（`sessions/*.jsonl`）：当前对话线程上下文。
- 每日记忆（`daily/*.jsonl`）：跨会话的当天信息聚合。
- 日志（`logs/llm_dialogue.jsonl`）：可追溯审计，记录每轮输入、上下文、输出、工具轨迹。

### 1.1 全局与用户技能配置（skills.yaml）

Agent 支持全局技能与按用户技能两层配置：

```text
skills/global/skills.yaml       # 全局默认技能
skills/{user_id}/skills.yaml    # 用户覆盖配置
```

全局示例（`skills/global/skills.yaml`）：

```yaml
skills:
  - name: memory-tools
    enabled: true
```

用户示例（`skills/{user_id}/skills.yaml`）：

```yaml
skills:
  - name: memory-tools
    enabled: false
  - name: planner
    enabled: true
```

规则：
- 全局配置与用户配置会合并；同名技能下，用户配置优先。
- 最终仅加载 `enabled: true` 的技能。
- 全局技能目录：`.sandbox/skills/global/{name}/SKILL.md`。
- 用户技能目录：`.sandbox/skills/{user_id}/{name}/SKILL.md`。
- 技能目录中必须存在 `SKILL.md` 才会加载。
- 文件不存在或格式错误时会自动降级为“不加载任何技能”，不影响聊天主流程。

### 1.2 技能文件 Shell 工具边界

Agent 提供 `bash_command` 用于读取/修改技能目录下文件。此工具是受限 shell：

- 统一使用 `bash -lc` 执行命令，工作目录固定为项目根目录。
- 路径表达约定：`root/` 表示工作区根目录。
- 内置 sandbox 为 `.sandbox/`（技能默认放在 `.sandbox/skills/`）。
- 若配置了 `config/sandbox.yaml` 的 `sandbox.writable_dir`，也可以访问该目录（使用其真实路径或 `root/` 相对路径）。
- 禁止绝对路径、`..`、管道、重定向、多命令拼接、环境变量插值。

常用正确示例：

```bash
ls .sandbox/skills/demo-user
cat .sandbox/skills/demo-user/demo-user-style/SKILL.md
cat root/README.md
mkdir -p .sandbox/skills/demo-user/new-skill
```

错误示例：

- `cat ../skills/demo-user/demo-user-style/SKILL.md`（包含 `..`）
- `cat /etc/passwd`（绝对路径）
- `cat .sandbox/skills/demo-user/demo-user-style/SKILL.md ; ls`（多命令）

### 2. 统一消息结构

`session` 与 `daily` 均使用 JSONL 单行消息，字段统一：

```json
{
  "ts": "2026-04-08T10:00:00",
  "role": "assistant",
  "content": "回复内容",
  "tags": ["session"],
  "meta": {
    "tool_calls": [
      {
        "name": "search_session",
        "input": "找工作",
        "output": "#11 ..."
      }
    ]
  },
  "message_id": 12
}
```

关键点：
- `message_id` 在单个文件内单调递增（从 1 开始），作为检索锚点。
- 历史老数据若缺失 `message_id`，读取时会自动标准化补齐。
- assistant 消息会持久化 `tool_calls` 到 `meta`，前端刷新后仍可回放工具调用。

### 3. 检索策略：锚点 + 窗口

核心不是“搜到一条就结束”，而是两段式：

1. 搜索：
- `search_session(keyword)`
- `search_daily(keyword, day)`

返回命中消息及 `message_id`。

2. 扩展：
- `get_session_window(message_id, before, after)`
- `get_daily_window(message_id, before, after, day)`

围绕锚点拉取前后文，解决“答案在下一条 / 上一条”的问题。

### 4. 上下文组装策略

当前 `chat` 默认上下文策略：
- 注入最近 3 轮 session 摘要（实现中按 6 条消息）
- 默认不直接注入 daily（`daily_limit=0`），由 Agent 按需调用工具检索
- 长期记忆来自 `memory.md` 全文

Agent 系统提示词明确要求：
- 优先用工具，能调用就调用
- 只有工具确实无法解决时才允许不调用
- 对可验证问题禁止猜测回答

### 5. 可观测性与前端回放

每轮 `/api/chat` 返回：
- `reply`
- `memory_hits`
- `title`
- `tool_calls`

同时 `tool_calls` 会持久化：
- 写入 session/daily 消息 `meta.tool_calls`
- 写入 `logs/llm_dialogue.jsonl`

历史加载 `/api/conversations/{session_id}/messages` 时会返回每条消息的 `tool_calls`，前端展示为可折叠工具面板，支持刷新后继续查看。

## 核心流程（单轮对话）

1. 用户消息写入 `session` + `daily`
2. 读取并构建 memory context
3. Agent 按提示词优先调用检索工具
4. 生成回复 + 工具轨迹
5. assistant 消息（含工具轨迹）写入 `session` + `daily`
6. 追加 `llm_dialogue` 日志
7. 更新 `conversations.json` 的 `updated_at`

## API

- `POST /api/chat`
  - 入参：`user_id`, `session_id`, `message`
  - 出参：`reply`, `memory_hits`, `title`, `tool_calls`

- `POST /api/conversations`
  - 入参：`user_id`
  - 出参：`session_id`

- `GET /api/conversations?user_id=...`
  - 返回会话列表（来源于 `conversations.json`，按 `updated_at` 倒序）

- `GET /api/conversations/{session_id}/messages?user_id=...`
  - 返回会话消息历史（含 `message_id` 与 `tool_calls`）

## 快速开始

1. 安装依赖

```bash
python -m pip install -e .[dev]
```

2. 配置 LLM

编辑 [config/llm.yaml](config/llm.yaml)：

```yaml
llm:
  api_key: "your_key"
  model: "gpt-4o-mini"
  base_url: "https://api.openai.com/v1"
```

`api_key` 为空时，系统会使用 fallback 回复逻辑，便于本地验证 memory 流程。

3. 启动服务

```bash
python -m uvicorn wozclaw.app:app --reload
```

4. 打开页面

```text
http://127.0.0.1:8000
```

## 测试

```bash
pytest -q
```

## 适合继续扩展的方向

- 将长期记忆从全文注入升级为结构化条目 + 检索排序。
- 增加“记忆冲突检测”与合并策略（偏好变更追踪）。
- 将工具调用日志做成可筛选时间线（按工具名/关键词过滤）。
- 引入向量召回并与当前关键词检索做 hybrid rerank。
