# WozClaw Design

## 1. 目标
构建一个最小可运行的对话 Agent Demo，支持文件系统记忆，能记住用户长期偏好、会话短期上下文和每日记忆，并通过前端页面交互验证效果。

## 2. 范围
- 使用 AgentScope 搭建最小 ReAct Agent。
- 文件系统记忆目录按用户隔离。
- 提供后端 API 与简易前端。
- 提供最基础测试覆盖核心记忆行为。

非目标：
- 不做生产级权限体系。
- 不做复杂向量数据库。
- 不做多模型路由。

## 3. 记忆模型
存储根目录：memory/

按用户隔离：memory/{user_id}/
- long_term.jsonl：长期记忆，结构化 note（例如偏好、事实、稳定约束）
- daily/YYYY-MM-DD.jsonl：每天的对话片段和摘要
- sessions/{session_id}.jsonl：会话级短期记忆

每条记录统一字段：
- ts: ISO 时间戳
- role: user/assistant/system
- content: 文本内容
- tags: 标签数组
- meta: 扩展对象（例如 topic、source）

## 4. 读取策略（混合）
1. 先规则读取：当前 session 最近 N 条 + 今日 daily 最近 M 条。
2. 再长期召回：对 long_term 做轻量关键词匹配（token overlap）选 top_k。
3. 合并去重后组装成 agent 上下文。

## 5. Agent 设计
- 主体：AgentScope 的 ReActAgent。
- Tool:
  - remember_note(note, tags): 将长期记忆写入 long_term.jsonl。
- Model:
  - 默认使用 AgentScope 的 OpenAIChatModel（需 OPENAI_API_KEY）。
  - 若缺少 key，回退到规则回复，保证 demo 可运行。

## 6. API 设计
- POST /api/chat
  - 入参：user_id, session_id, message
  - 流程：
    1) 写入 user 消息到 session/daily
    2) 读取 memory context
    3) 调用 agent 生成回复
    4) 写入 assistant 消息到 session/daily
    5) 返回 reply + memory_hits

- GET /
  - 返回前端页面。

## 7. 前端
- 单页聊天界面：user_id、session_id 输入框 + 消息列表 + 输入框。
- 展示 assistant 回复与命中的记忆摘要。

## 8. 错误处理
- user_id/session_id 非法时返回 400。
- 写文件失败返回 500 并携带可读错误。
- 模型不可用时回退规则回复而非硬失败。

## 9. 测试
- memory store:
  - 写入后文件存在且记录可读。
  - 读取策略能返回 session + daily + long_term 组合。
- service:
  - 缺少模型 key 时可用回退路径生成回复。

## 10. 验收标准
- 启动服务后可在页面多轮对话。
- 同 user + session 能记住上下文。
- 新会话仍能命中长期记忆。
- 测试通过。
