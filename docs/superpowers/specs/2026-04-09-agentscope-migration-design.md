# WozClaw AgentScope Migration Design

## 1. Goal
在不改变现有业务设计逻辑的前提下，将项目中的 Agent 运行时统一为 AgentScope。

## 2. Hard Constraints
- ReAct 逻辑保持不变。
- memory 分层、读写、检索与上下文组装策略保持不变。
- 前端 API 入参与返回字段保持不变。
- 保持现有 fallback 路径（无 key 时 deterministic 回复）。

## 3. Scope
### In Scope
- 统一 `src/wozclaw/agent.py` 为 AgentScope 运行时实现。
- 统一会话标题生成为 AgentScope 模型调用实现。
- 统一依赖声明为 AgentScope 技术栈。
- 必要的测试适配与新增回归测试。

### Out of Scope
- 不修改 `ChatService` 的接口和调用语义。
- 不修改 FastAPI 路由和前端协议。
- 不修改 memory 数据结构、存储路径和文件格式。

## 4. Architecture
采用“外部兼容 + 内核重构”方案：
- 对外继续保留 `ReActMemoryAgent`、`AgentResponse`、`ConversationTitleGenerator` 的接口语义。
- 对内改用 AgentScope 原生模型与工具执行路径。
- 工具定义仍为 6 个同名工具：`remember_note`、`get_recent_session`、`search_session`、`get_session_window`、`search_daily`、`get_daily_window`。
- `tool_calls` 输出结构保持 `name/input/output`。

## 5. Data Flow
保持现有链路不变：
1. user 消息写入 session/daily。
2. 读取并构建 memory_context。
3. Agent 基于系统提示词与工具进行 ReAct 处理。
4. assistant 回复与工具轨迹写入 session/daily/log。
5. 会话标题生成与会话更新时间刷新。

唯一变化：步骤 3 和标题生成的模型调用框架统一为 AgentScope。

## 6. Error Handling
- AgentScope 初始化失败：回退 FallbackAgent，不中断主流程。
- 工具执行失败：记录错误摘要到 `tool_calls.output`，并继续完成回答。
- 模型调用失败：回退 FallbackAgent。
- 标题生成失败：保持现有 fallback（截断用户消息）。

## 7. API Compatibility
以下字段保持完全一致：
- `/api/chat`: `reply`, `memory_hits`, `title`, `tool_calls`
- `/api/conversations/{session_id}/messages`: `message_id`, `tool_calls` 等历史消息结构

## 8. Testing Strategy
- 保持并通过现有测试：
  - `tests/test_agent_prompt.py`
  - `tests/test_chat_service.py`
  - `tests/test_app_api.py`
- 新增最小回归测试：
  - AgentScope 工具轨迹结构（`name/input/output`）
  - 无 key fallback 行为一致性
- 验证命令：`pytest -q`

## 9. Acceptance Criteria
- 代码库文档与实现统一采用 AgentScope 术语与依赖。
- ReAct + memory 行为保持现有逻辑。
- 前端 API 契约与字段完全不变。
- 全量测试通过。
