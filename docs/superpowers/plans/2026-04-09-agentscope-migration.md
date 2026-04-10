# AgentScope Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保持现有业务逻辑和 API 契约不变的前提下，将项目中的 Agent 运行时统一为 AgentScope。

**Architecture:** 保留 `ReActMemoryAgent` 与 `ConversationTitleGenerator` 的外部接口，内部运行时迁移到 AgentScope。`ChatService` 和 FastAPI 接口不变，tool_calls 结构继续输出 `name/input/output` 供前端回放。

**Tech Stack:** Python 3.11, FastAPI, AgentScope, Pytest

---

### Task 1: 增加迁移防回归测试（先红）

**Files:**
- Modify: `tests/test_agent_prompt.py`
- Test: `tests/test_agent_prompt.py`

- [ ] **Step 1: 写失败测试，验证不再依赖旧框架导入**

```python
from pathlib import Path


def test_agent_module_no_legacy_framework_imports() -> None:
    source = Path("src/agent_memory_demo/agent.py").read_text(encoding="utf-8")

    assert "legacy framework import marker" not in source
```

- [ ] **Step 2: 运行单测确认失败（RED）**

Run: `pytest -q tests/test_agent_prompt.py::test_agent_module_no_legacy_framework_imports`
Expected: FAIL，当前代码仍包含旧框架导入。

- [ ] **Step 3: 提交测试变更**

```bash
git add tests/test_agent_prompt.py
git commit -m "test: add migration guard against legacy framework imports"
```

### Task 2: 迁移 Agent 运行时到 AgentScope（绿）

**Files:**
- Modify: `src/agent_memory_demo/agent.py`
- Test: `tests/test_agent_prompt.py`

- [ ] **Step 1: 实现 AgentScope ReAct 运行时并保留现有外部接口**

```python
# 关键实现要点（完整代码在变更文件中实现）:
# 1) 删除旧框架导入
# 2) 新增 AgentScope 模型客户端与工具调用循环
# 3) 保持 ReActMemoryAgent.respond(...) -> AgentResponse(text, tool_calls)
# 4) 保留 build_system_prompt 内容
# 5) 工具轨迹统一输出: {"name": "", "input": "", "output": ""}
```

- [ ] **Step 2: 运行测试确认通过（GREEN）**

Run: `pytest -q tests/test_agent_prompt.py`
Expected: PASS。

- [ ] **Step 3: 提交 Agent 迁移变更**

```bash
git add src/agent_memory_demo/agent.py tests/test_agent_prompt.py
git commit -m "refactor: migrate runtime to agentscope"
```

### Task 3: 替换依赖并清理说明文档

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Test: `tests/test_llm_config.py`

- [ ] **Step 1: 依赖替换**

```toml
# 从 dependencies 删除:
# - legacy-framework-a
# - legacy-framework-b
# - legacy-framework-openai-adapter
# 新增:
# - agentscope>=<稳定版本>
```

- [ ] **Step 2: 文档替换技术栈表述**

```markdown
- AgentScope ReAct Agent
```

- [ ] **Step 3: 运行基础配置测试**

Run: `pytest -q tests/test_llm_config.py`
Expected: PASS。

- [ ] **Step 4: 提交依赖与文档变更**

```bash
git add pyproject.toml README.md
git commit -m "chore: align dependencies with agentscope"
```

### Task 4: 全量回归验证与收尾

**Files:**
- Test: `tests/test_agent_prompt.py`
- Test: `tests/test_chat_service.py`
- Test: `tests/test_app_api.py`
- Test: `tests/test_memory_store.py`
- Test: `tests/test_frontend_rendering.py`

- [ ] **Step 1: 运行全量测试**

Run: `pytest -q`
Expected: 全部 PASS。

- [ ] **Step 2: 全库字符串扫描，确认无旧框架运行时残留**

Run: `rg -n "legacy-framework-a|legacy-framework-b" src tests pyproject.toml README.md`
Expected: 仅允许在历史文档或迁移说明中出现，业务运行时代码无残留。

- [ ] **Step 3: 最终提交**

```bash
git add src pyproject.toml README.md tests
git commit -m "feat: migrate project runtime to agentscope"
```
