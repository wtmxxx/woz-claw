from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from agent_memory_demo.agent import AgentResponse, ReActMemoryAgent, ConversationTitleGenerator
from agent_memory_demo.memory_store import MemoryStore


@dataclass
class ChatResult:
    reply: str
    memory_hits: int
    title: str
    tool_calls: list[dict[str, str]] = field(default_factory=list)
    loaded_skills: list[dict[str, str]] = field(default_factory=list)
    activity_traces: list[dict[str, str]] = field(default_factory=list)


class ChatService:
    def __init__(
        self,
        memory_store: MemoryStore,
        agent: object | None = None,
        title_generator: object | None = None,
    ) -> None:
        self.memory_store = memory_store
        self._agent = agent
        self._title_generator = title_generator

    def chat(self, user_id: str, session_id: str, message: str) -> ChatResult:
        self._validate_ids(user_id, session_id)

        self.memory_store.append_session_message(
            user_id, session_id, "user", message)
        self.memory_store.append_daily_message(user_id, "user", message)

        context = self.memory_store.load_context(
            user_id,
            session_id,
            query=message,
            session_limit=6,
            daily_limit=0,
        )
        prompt_context = self.memory_store.build_prompt_context(context)

        if self._agent is None:
            runtime_agent = ReActMemoryAgent(
                memory_store=self.memory_store,
                user_id=user_id,
                session_id=session_id,
            )
            agent_result = runtime_agent.respond(message, prompt_context)
        else:
            raw_result = self._agent.respond(message, prompt_context)
            if isinstance(raw_result, AgentResponse):
                agent_result = raw_result
            else:
                agent_result = AgentResponse(
                    text=str(raw_result),
                    tool_calls=[],
                    loaded_skills=[],
                    activity_traces=[],
                )

        reply = agent_result.text
        tool_calls = agent_result.tool_calls
        loaded_skills = agent_result.loaded_skills
        activity_traces = agent_result.activity_traces
        assistant_meta: dict[str, Any] = {}
        if tool_calls:
            assistant_meta["tool_calls"] = tool_calls
        if loaded_skills:
            assistant_meta["loaded_skills"] = loaded_skills
        if activity_traces:
            assistant_meta["activity_traces"] = activity_traces

        self.memory_store.append_session_message(
            user_id, session_id, "assistant", reply, meta=assistant_meta)
        self.memory_store.append_daily_message(
            user_id,
            "assistant",
            reply,
            meta=assistant_meta,
        )
        self.memory_store.append_llm_dialogue_log(
            user_id=user_id,
            session_id=session_id,
            user_message=message,
            assistant_reply=reply,
            memory_context=prompt_context,
            tool_calls=tool_calls,
        )

        title = self.memory_store.get_conversation_title(user_id, session_id)
        if not title:
            if self._title_generator is None:
                generator = ConversationTitleGenerator()
                title = generator.generate_title(message, reply)
            else:
                title = self._title_generator.generate_title(message, reply)
        # Refresh updated_at every round so conversation list stays sorted by latest activity.
        self.memory_store.set_conversation_title(
            user_id, session_id, title)

        return ChatResult(
            reply=reply,
            memory_hits=len(context.long_term_hits),
            title=title,
            tool_calls=tool_calls,
            loaded_skills=loaded_skills,
            activity_traces=activity_traces,
        )

    def create_conversation(self, user_id: str) -> str:
        self._validate_ids(user_id, "seed")
        return uuid4().hex[:12]

    def list_conversations(self, user_id: str) -> list[dict[str, str]]:
        self._validate_ids(user_id, "seed")
        return self.memory_store.list_conversations(user_id)

    def get_session_messages(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        self._validate_ids(user_id, session_id)
        rows = self.memory_store.get_session_messages(user_id, session_id)
        result: list[dict[str, Any]] = []
        for row in rows:
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            raw_tool_calls = meta.get(
                "tool_calls") if isinstance(meta, dict) else []
            tool_calls: list[dict[str, str]] = []
            if isinstance(raw_tool_calls, list):
                for item in raw_tool_calls:
                    if not isinstance(item, dict):
                        continue
                    tool_calls.append(
                        {
                            "name": str(item.get("name", "")),
                            "input": str(item.get("input", "")),
                            "output": str(item.get("output", "")),
                        }
                    )

            raw_loaded_skills = meta.get(
                "loaded_skills") if isinstance(meta, dict) else []
            loaded_skills: list[dict[str, str]] = []
            if isinstance(raw_loaded_skills, list):
                for item in raw_loaded_skills:
                    if not isinstance(item, dict):
                        continue
                    loaded_skills.append(
                        {
                            "name": str(item.get("name", "")),
                            "source": str(item.get("source", "")),
                            "dir": str(item.get("dir", "")),
                        }
                    )

            raw_activity_traces = meta.get(
                "activity_traces") if isinstance(meta, dict) else []
            activity_traces: list[dict[str, str]] = []
            if isinstance(raw_activity_traces, list):
                for item in raw_activity_traces:
                    if not isinstance(item, dict):
                        continue
                    activity_traces.append(
                        {
                            "type": str(item.get("type", "")),
                            "name": str(item.get("name", "")),
                            "source": str(item.get("source", "")),
                            "dir": str(item.get("dir", "")),
                            "input": str(item.get("input", "")),
                            "output": str(item.get("output", "")),
                        }
                    )

            result.append(
                {
                    "ts": str(row.get("ts", "")),
                    "role": str(row.get("role", "")),
                    "content": str(row.get("content", "")),
                    "message_id": int(row.get("message_id", 0)) if str(row.get("message_id", "")).isdigit() else 0,
                    "tool_calls": tool_calls,
                    "loaded_skills": loaded_skills,
                    "activity_traces": activity_traces,
                }
            )
        return result

    def _validate_ids(self, user_id: str, session_id: str) -> None:
        if not user_id.strip() or not session_id.strip():
            raise ValueError("user_id and session_id are required")
