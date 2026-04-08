from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agent_memory_demo.config import load_llm_config
from agent_memory_demo.memory_store import MemoryStore


@dataclass
class AgentResponse:
    text: str
    tool_calls: list[dict[str, str]] = field(default_factory=list)


class FallbackAgent:
    def respond(self, user_message: str, memory_context: str) -> str:
        # Keep fallback deterministic for local demo and tests.
        if "中文" in memory_context or "language" in memory_context.lower():
            return f"收到：{user_message}。我会继续用中文回答。"
        return f"收到：{user_message}。我已经记录到记忆中。"


class ReActMemoryAgent:
    def __init__(self, memory_store: MemoryStore, user_id: str, session_id: str) -> None:
        self.memory_store = memory_store
        self.user_id = user_id
        self.session_id = session_id

        llm_config = load_llm_config()
        if not llm_config.api_key:
            self._graph = None
            self._fallback = FallbackAgent()
            return

        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            api_key=llm_config.api_key,
            model=llm_config.model,
            base_url=llm_config.base_url or None,
            temperature=0.2,
        )

        @tool
        def remember_note(note: str, tags: str = "preference,fact") -> str:
            """Overwrite memory.md with the latest full-memory text. The input must be complete, not incremental."""
            tag_list = [item.strip()
                        for item in tags.split(",") if item.strip()]
            self.memory_store.remember_long_term(
                self.user_id, note, tags=tag_list)
            return "stored"

        @tool
        def get_recent_session(rounds: int = 3) -> str:
            """Fetch the latest session history by rounds (1 round ~= user+assistant messages)."""
            rows = self.memory_store.get_recent_session_messages(
                self.user_id, self.session_id, rounds=max(1, rounds))
            if not rows:
                return "no session history"
            return "\n".join(f"#{row.get('message_id', '')} {row.get('ts', '')} | {row.get('role', '')}: {row.get('content', '')}" for row in rows)

        @tool
        def search_session(keyword: str, limit: int = 20) -> str:
            """Search session history by keyword and return message_id anchors for follow-up window lookups."""
            rows = self.memory_store.search_session_messages(
                self.user_id,
                self.session_id,
                keyword=keyword,
                limit=max(1, min(limit, 100)),
            )
            if not rows:
                return "no matched session messages"
            return "\n".join(f"#{row.get('message_id', '')} {row.get('ts', '')} | {row.get('role', '')}: {row.get('content', '')}" for row in rows)

        @tool
        def get_session_window(message_id: int, before: int = 0, after: int = 0) -> str:
            """Fetch session messages around a message_id anchor. Use after search_session to expand context."""
            rows = self.memory_store.get_session_messages_window(
                self.user_id,
                self.session_id,
                message_id=message_id,
                before=max(0, before),
                after=max(0, after),
            )
            if not rows:
                return "no window messages"
            return "\n".join(f"#{row.get('message_id', '')} {row.get('ts', '')} | {row.get('role', '')}: {row.get('content', '')}" for row in rows)

        @tool
        def search_daily(keyword: str, day: str = "", limit: int = 50) -> str:
            """Search daily history by keyword. Optional day=YYYY-MM-DD; empty/invalid day falls back to all days."""
            rows = self.memory_store.search_daily_messages(
                self.user_id,
                keyword=keyword,
                limit=max(1, min(limit, 200)),
                day=day,
            )
            if not rows:
                return "no matched daily messages"
            return "\n".join(f"#{row.get('message_id', '')} {row.get('ts', '')} | {row.get('role', '')}: {row.get('content', '')}" for row in rows)

        @tool
        def get_daily_window(message_id: int, before: int = 0, after: int = 0, day: str = "") -> str:
            """Fetch daily messages around a message_id anchor. Use after search_daily to expand context."""
            rows = self.memory_store.get_daily_messages_window(
                self.user_id,
                message_id=message_id,
                before=max(0, before),
                after=max(0, after),
                day=day,
            )
            if not rows:
                return "no window messages"
            return "\n".join(f"#{row.get('message_id', '')} {row.get('ts', '')} | {row.get('role', '')}: {row.get('content', '')}" for row in rows)

        self._graph = create_react_agent(
            llm,
            tools=[remember_note, get_recent_session,
                   search_session, get_session_window, search_daily, get_daily_window],
        )
        self._fallback = FallbackAgent()

    def respond(self, user_message: str, memory_context: str) -> AgentResponse:
        if self._graph is None:
            return AgentResponse(text=self._fallback.respond(user_message, memory_context), tool_calls=[])

        prompt = self.build_system_prompt(memory_context)

        result = self._graph.invoke(
            {
                "messages": [
                    SystemMessage(content=prompt),
                    HumanMessage(content=user_message),
                ]
            }
        )

        messages = result.get("messages", [])
        if not messages:
            return AgentResponse(text=self._fallback.respond(user_message, memory_context), tool_calls=[])
        return AgentResponse(
            text=str(messages[-1].content),
            tool_calls=self._extract_tool_calls(messages),
        )

    def _extract_tool_calls(self, messages: list[Any]) -> list[dict[str, str]]:
        traces: list[dict[str, str]] = []

        for message in messages:
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls is None:
                additional_kwargs = getattr(message, "additional_kwargs", None)
                if isinstance(additional_kwargs, dict):
                    tool_calls = additional_kwargs.get("tool_calls")
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    function_part = call.get("function") if isinstance(
                        call.get("function"), dict) else {}
                    name = str(call.get("name")
                               or function_part.get("name") or "")
                    args = call.get("args", call.get(
                        "arguments", function_part.get("arguments", "")))
                    traces.append(
                        {
                            "name": name,
                            "input": self._stringify_tool_value(args),
                            "output": "",
                        }
                    )

            if message.__class__.__name__ != "ToolMessage":
                continue

            tool_name = str(getattr(message, "name", "") or "")
            tool_output = self._stringify_tool_value(
                getattr(message, "content", ""))

            if tool_name:
                pending = next(
                    (
                        item
                        for item in reversed(traces)
                        if item.get("name") == tool_name and not item.get("output")
                    ),
                    None,
                )
                if pending is not None:
                    pending["output"] = tool_output
                    continue

            traces.append({"name": tool_name or "tool",
                          "input": "", "output": tool_output})

        return traces

    def _stringify_tool_value(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    def build_system_prompt(self, memory_context: str) -> str:
        return (
            "你是一个带记忆的助手。你会优先遵守长期记忆中的稳定偏好。\n"
            "当用户信息有新增或变化时，你可以调用 remember_note。\n"
            "默认只给你最近3轮 session 摘要；如果需要更早的 session 或 daily 历史，必须主动调用搜索工具检索。\n"
            "工具使用总原则：优先调用工具，能调用工具就调用工具；只要工具可以提供事实依据，就必须先调用再回答。\n"
            "只有在工具确实无法解决问题时，才允许不调用工具并直接给出说明。\n"
            "禁止仅凭猜测回答可被工具验证的问题。\n"
            "search_session 和 search_daily 会返回 message_id；如果命中内容前后还需要更多上下文，使用 get_session_window(message_id, before, after) 扩展。\n"
            "daily 也支持同样的窗口扩展：get_daily_window(message_id, before, after, day)。\n"
            "关键规则：remember_note 的 note 必须是 memory.md 的完整版本，不是增量补丁。\n"
            "你写入的完整版本必须保留旧信息（仍然有效的部分）并融合最新变化，避免只写最新一条导致旧特征丢失。\n"
            "如果某条旧偏好被新信息明确否定，要在完整版本中替换该条而不是并存冲突表述。\n\n"
            f"可用记忆上下文:\n{memory_context}"
        )


class ConversationTitleGenerator:
    def __init__(self) -> None:
        self._config = load_llm_config()

    def generate_title(self, user_message: str, assistant_reply: str) -> str:
        if not self._config.api_key:
            return self._fallback(user_message)

        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                api_key=self._config.api_key,
                model=self._config.model,
                base_url=self._config.base_url or None,
                temperature=0,
            )
            prompt = (
                "请根据以下对话生成一个简短中文会话标题（不超过12字，不要标点和引号）。\n"
                f"用户: {user_message}\n"
                f"助手: {assistant_reply}\n"
                "只输出标题本身。"
            )
            text = llm.invoke(prompt).content
            title = str(text).strip().replace("\n", "")
            return title[:12] if title else self._fallback(user_message)
        except Exception:
            return self._fallback(user_message)

    def _fallback(self, user_message: str) -> str:
        base = user_message.strip() or "新对话"
        return base[:12]
