# Agent Memory Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个可运行的文件系统 memory + AgentScope ReAct agent + 前端交互 demo。

**Architecture:** FastAPI 提供 API 与静态页面；MemoryStore 负责按 user/session/day 写读文件；ChatService 组合记忆上下文并驱动 agent；agent 支持 OpenAI 模型与无 key 回退。

**Tech Stack:** Python 3.11+, FastAPI, AgentScope, pytest

---

### Task 1: MemoryStore 测试与实现

**Files:**
- Create: `tests/test_memory_store.py`
- Create: `src/agent_memory_demo/memory_store.py`

- [ ] Step 1: 写 failing tests
- [ ] Step 2: 运行测试验证失败
- [ ] Step 3: 实现最小 MemoryStore
- [ ] Step 4: 运行测试验证通过

### Task 2: Agent 与 Service

**Files:**
- Create: `src/agent_memory_demo/agent.py`
- Create: `src/agent_memory_demo/service.py`
- Create: `tests/test_chat_service.py`

- [ ] Step 1: 写 service 回退逻辑 failing test
- [ ] Step 2: 运行失败验证
- [ ] Step 3: 实现 agent 封装与 service
- [ ] Step 4: 测试通过

### Task 3: FastAPI 与前端

**Files:**
- Create: `src/agent_memory_demo/app.py`
- Create: `src/agent_memory_demo/static/index.html`
- Create: `src/agent_memory_demo/__init__.py`

- [ ] Step 1: 添加 API 与静态页面
- [ ] Step 2: 本地启动验证可访问

### Task 4: 文档与运行说明

**Files:**
- Create: `README.md`

- [ ] Step 1: 添加安装与启动说明
- [ ] Step 2: 添加 memory 目录结构说明
