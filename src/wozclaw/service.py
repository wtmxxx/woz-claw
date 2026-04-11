from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any
from uuid import uuid4

from wozclaw.agent import AgentResponse, ReActMemoryAgent, ConversationTitleGenerator
from wozclaw.memory_store import MemoryStore


@dataclass
class ChatResult:
    reply: str
    memory_hits: int
    title: str
    tool_calls: list[dict[str, str]] = field(default_factory=list)
    loaded_skills: list[dict[str, str]] = field(default_factory=list)
    activity_traces: list[dict[str, str]] = field(default_factory=list)
    approval_request: dict[str, Any] | None = None
    choice_request: dict[str, Any] | None = None


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
        self._pending_runtime_agents: dict[str, ReActMemoryAgent] = {}

    def chat(self, user_id: str, session_id: str, message: str) -> ChatResult:
        self._validate_ids(user_id, session_id)

        self.memory_store.append_session_message(
            user_id, session_id, "user", message)
        self.memory_store.append_daily_message(user_id, "user", message)

        context = self.memory_store.load_context(
            user_id,
            session_id,
            query=message,
            session_token_budget=2800,
            daily_limit=0,
        )
        prompt_context = self.memory_store.build_prompt_context(context)

        runtime_agent: ReActMemoryAgent | None = None
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
        raw_tool_calls = agent_result.tool_calls
        tool_calls = self._filter_placeholder_tool_calls(raw_tool_calls)
        loaded_skills = agent_result.loaded_skills
        raw_activity_traces = agent_result.activity_traces
        activity_traces = self._filter_placeholder_activity_traces(
            raw_activity_traces)
        persisted_tool_calls = tool_calls
        persisted_activity_traces = activity_traces
        approval_request = self._extract_approval_request(raw_tool_calls)
        choice_request = self._extract_choice_request(raw_tool_calls)
        assistant_meta: dict[str, Any] = {}
        if persisted_tool_calls:
            assistant_meta["tool_calls"] = persisted_tool_calls
        if loaded_skills:
            assistant_meta["loaded_skills"] = loaded_skills
        if persisted_activity_traces:
            assistant_meta["activity_traces"] = persisted_activity_traces

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

        if approval_request and str(approval_request.get("request_id", "")).strip():
            req_id = str(approval_request.get("request_id", "")).strip()
            if runtime_agent is not None:
                self._pending_runtime_agents[req_id] = runtime_agent
            self.memory_store.set_pending_react_state(
                user_id,
                session_id,
                req_id,
                {
                    "user_message": message,
                    "prompt_context": prompt_context,
                },
            )

        if choice_request and str(choice_request.get("request_id", "")).strip():
            self.memory_store.create_pending_choice(
                user_id,
                session_id,
                choice_request,
            )

        title = self.memory_store.get_conversation_title(user_id, session_id)
        if not title:
            if self._title_generator is None:
                generator = ConversationTitleGenerator()
                title = generator.generate_title(message, "")
            else:
                title = self._title_generator.generate_title(message, "")
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
            approval_request=approval_request,
            choice_request=choice_request,
        )

    def resume_after_approval(
        self,
        user_id: str,
        session_id: str,
        request_id: str,
        command: str,
        output: str,
        approved: bool,
    ) -> ChatResult:
        self._validate_ids(user_id, session_id)
        req_id = request_id.strip()
        self.memory_store.replace_approval_placeholder_output(
            user_id,
            session_id,
            req_id,
            output,
        )
        state = self.memory_store.pop_pending_react_state(
            user_id, session_id, req_id)
        runtime_agent = self._pending_runtime_agents.pop(req_id, None)

        initial_tool_call = {
            "name": "bash_command",
            "input": command,
            "output": output,
        }

        if state is None and runtime_agent is None:
            reply = f"已{'批准' if approved else '拒绝'}执行命令\n$ {command}\n\n{output}"
            meta = {"tool_calls": [initial_tool_call]}
            self.memory_store.append_session_message(
                user_id, session_id, "assistant", reply, meta=meta)
            self.memory_store.append_daily_message(
                user_id, "assistant", reply, meta=meta)
            title = self.memory_store.get_conversation_title(
                user_id, session_id) or "新对话"
            self.memory_store.set_conversation_title(
                user_id, session_id, title)
            return ChatResult(
                reply=reply,
                memory_hits=0,
                title=title,
                tool_calls=[initial_tool_call],
            )

        user_message = str(state.get("user_message", "")
                           ).strip() if isinstance(state, dict) else ""
        if not user_message:
            user_message = "继续当前任务"
        resume_message = (
            f"继续刚才被审批中断的任务。命令 `{command}` 已批准执行，输出如下：\n{output}\n"
            if approved else
            f"继续刚才被审批中断的任务。命令 `{command}` 被用户拒绝执行。请在此约束下继续任务。"
        )

        context = self.memory_store.load_context(
            user_id,
            session_id,
            query=user_message,
            session_token_budget=2800,
            daily_limit=0,
        )
        prompt_context = self.memory_store.build_prompt_context(context)

        if self._agent is None:
            if runtime_agent is None:
                runtime_agent = ReActMemoryAgent(
                    memory_store=self.memory_store,
                    user_id=user_id,
                    session_id=session_id,
                )
            agent_result = runtime_agent.respond(
                resume_message, prompt_context)
        else:
            raw_result = self._agent.respond(resume_message, prompt_context)
            if isinstance(raw_result, AgentResponse):
                agent_result = raw_result
            else:
                agent_result = AgentResponse(text=str(raw_result))

        raw_tool_calls = [initial_tool_call, *agent_result.tool_calls]
        tool_calls = self._filter_placeholder_tool_calls(raw_tool_calls)
        loaded_skills = agent_result.loaded_skills
        initial_activity_trace = {
            "type": "tool",
            "name": "bash_command",
            "source": "",
            "dir": "",
            "input": command,
            "output": output,
        }
        raw_activity_traces = [initial_activity_trace,
                               *agent_result.activity_traces]
        activity_traces = self._filter_placeholder_activity_traces(
            raw_activity_traces)
        approval_request = self._extract_approval_request(raw_tool_calls)
        choice_request = self._extract_choice_request(raw_tool_calls)
        reply = agent_result.text or "已处理审批并继续执行。"

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
            user_id, "assistant", reply, meta=assistant_meta)
        self.memory_store.append_llm_dialogue_log(
            user_id=user_id,
            session_id=session_id,
            user_message=f"[APPROVAL_RESUME] {resume_message}",
            assistant_reply=reply,
            memory_context=prompt_context,
            tool_calls=raw_tool_calls,
        )

        if approval_request and str(approval_request.get("request_id", "")).strip():
            next_req_id = str(approval_request.get("request_id", "")).strip()
            if runtime_agent is not None:
                self._pending_runtime_agents[next_req_id] = runtime_agent
            self.memory_store.set_pending_react_state(
                user_id,
                session_id,
                next_req_id,
                {
                    "user_message": user_message,
                    "prompt_context": prompt_context,
                },
            )

        if choice_request and str(choice_request.get("request_id", "")).strip():
            self.memory_store.create_pending_choice(
                user_id,
                session_id,
                choice_request,
            )

        title = self.memory_store.get_conversation_title(user_id, session_id)
        if not title:
            if self._title_generator is None:
                title = ConversationTitleGenerator().generate_title(user_message, "")
            else:
                title = self._title_generator.generate_title(
                    user_message, "")
        self.memory_store.set_conversation_title(user_id, session_id, title)

        return ChatResult(
            reply=reply,
            memory_hits=len(context.long_term_hits),
            title=title,
            tool_calls=tool_calls,
            loaded_skills=loaded_skills,
            activity_traces=activity_traces,
            approval_request=approval_request,
            choice_request=choice_request,
        )

    def _filter_placeholder_tool_calls(self, tool_calls: list[dict[str, str]]) -> list[dict[str, str]]:
        marker = "__APPROVAL_REQUIRED__"
        result: list[dict[str, str]] = []
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", ""))
            output = str(item.get("output", ""))
            if name == "bash_command" and output.startswith(marker):
                continue
            result.append(item)
        return result

    def _filter_placeholder_activity_traces(self, traces: list[dict[str, str]]) -> list[dict[str, str]]:
        marker = "__APPROVAL_REQUIRED__"
        result: list[dict[str, str]] = []
        for item in traces:
            if not isinstance(item, dict):
                continue
            trace_type = str(item.get("type", ""))
            output = str(item.get("output", ""))
            if trace_type == "tool" and output.startswith(marker):
                continue
            result.append(item)
        return result

    def _extract_approval_request(self, tool_calls: list[dict[str, str]]) -> dict[str, Any] | None:
        marker = "__APPROVAL_REQUIRED__"
        for item in reversed(tool_calls):
            if str(item.get("name", "")) != "bash_command":
                continue
            output = str(item.get("output", ""))
            if not output.startswith(marker):
                continue
            raw = output[len(marker):].strip()
            try:
                parsed = json.loads(raw)
            except Exception:
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    def _extract_choice_request(self, tool_calls: list[dict[str, str]]) -> dict[str, Any] | None:
        marker = "__CHOICE_REQUIRED__"
        for item in reversed(tool_calls):
            if str(item.get("name", "")) != "ask_human_choice":
                continue
            output = str(item.get("output", ""))
            if not output.startswith(marker):
                continue
            raw = output[len(marker):].strip()
            try:
                parsed = json.loads(raw)
            except Exception:
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

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
