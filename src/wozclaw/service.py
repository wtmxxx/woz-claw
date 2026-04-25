from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
import json
from typing import Any, AsyncGenerator
from uuid import uuid4

from wozclaw.agent import (
    AgentResponse,
    ConversationTitleGenerator,
    ReActMemoryAgent,
)
from wozclaw.memory_store import MemoryStore


SESSION_MEMORY_COMPACT_THRESHOLD_TOKENS = 30000
SESSION_MEMORY_COMPACT_TARGET_TOKENS = 8000


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


class ApprovalInterrupt(Exception):
    """Raised to stop the current ReAct loop immediately when human approval is required."""


class ChoiceInterrupt(Exception):
    """Raised to stop the current ReAct loop immediately when human choice is required."""


class FallbackAgent:
    def respond(self, user_message: str, memory_context: str) -> str:
        if "中文" in memory_context or "language" in memory_context.lower():
            return f"收到：{user_message}。我会继续用中文回答。"
        return f"收到：{user_message}。我已经记录到记忆中。"


class LLMDialogueRecorder:
    """Wraps OpenAIChatModel to record each LLM call to dialogue log."""

    def __init__(
        self,
        model: Any,
        memory_store: MemoryStore,
        user_id: str,
        session_id: str,
        compact_model: Any | None = None,
        compact_threshold_tokens: int = SESSION_MEMORY_COMPACT_THRESHOLD_TOKENS,
    ) -> None:
        self._model = model
        self._memory_store = memory_store
        self._user_id = user_id
        self._session_id = session_id
        self._node_counter = 0
        self._compact_model = compact_model
        self._compact_threshold_tokens = max(1, int(compact_threshold_tokens))
        self._auto_tool_traces: list[dict[str, str]] = []

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped model."""
        return getattr(self._model, name)

    async def __call__(self, prompt: Any, **kwargs: Any) -> Any:
        """Intercept model calls to record raw LLM input/output only."""
        await self._auto_compact_session_memory_if_needed()
        filtered_prompt = self._filter_react_messages(prompt)
        response = await self._model(filtered_prompt, **kwargs)
        try:
            self._node_counter += 1
            assistant_text = self._extract_assistant_text(response)
            if assistant_text:
                self._memory_store.append_session_memory_message(
                    self._user_id,
                    self._session_id,
                    "assistant",
                    assistant_text,
                )
            request_envelope = {
                "messages": filtered_prompt,
            }
            if kwargs:
                request_envelope["kwargs"] = kwargs
            self._memory_store.append_llm_dialogue_log(
                user_id=self._user_id,
                session_id=self._session_id,
                input_value=self._serialize_value(request_envelope),
                output_value=self._serialize_value(response),
            )
        except Exception:
            pass
        return response

    def _filter_react_messages(self, prompt: Any) -> Any:
        if not isinstance(prompt, list):
            return prompt

        system_msg = None
        session_memory_msg = None
        latest_user_msg = None

        for item in prompt:
            role = self._message_field(item, "role")
            name = self._message_field(item, "name")
            if role == "system" and system_msg is None:
                system_msg = item
                continue

            if role != "user":
                continue

            if name == "session_memory":
                session_memory_msg = item
                continue

            if name == "user" or name is None:
                latest_user_msg = item

        latest_session_memory = self._load_latest_session_memory()
        if latest_session_memory:
            session_memory_msg = {
                "role": "user",
                "name": "session_memory",
                "content": [{"type": "text", "text": latest_session_memory}],
            }

        filtered: list[Any] = []
        if system_msg is not None:
            filtered.append(system_msg)
        if session_memory_msg is not None:
            filtered.append(session_memory_msg)
        if latest_user_msg is not None:
            filtered.append(latest_user_msg)

        if filtered:
            return filtered
        return prompt

    def _load_latest_session_memory(self) -> str:
        try:
            state = self._memory_store.get_session_state(
                self._user_id,
                self._session_id,
            )
            return str(state.get("session_memory", "")).strip()
        except Exception:
            return ""

    async def _auto_compact_session_memory_if_needed(self) -> None:
        if self._compact_model is None:
            return

        try:
            state = self._memory_store.get_session_state(
                self._user_id,
                self._session_id,
            )
            raw_memory = str(state.get("session_memory", "")).strip()
            if not raw_memory:
                return

            token_count = self._estimate_text_tokens(raw_memory)
            if token_count <= self._compact_threshold_tokens:
                return

            compacted = await self._compact_session_memory(raw_memory)
            compacted_text = compacted.strip()
            if not compacted_text:
                return

            self._memory_store.update_session_state(
                self._user_id,
                self._session_id,
                {"session_memory": compacted_text},
            )
            self._memory_store.append_session_memory_tool_trace(
                self._user_id,
                self._session_id,
                "compact_context",
                "",
                "上下文压缩成功",
            )
            self._auto_tool_traces.append(
                {
                    "name": "compact_context",
                    "input": "",
                    "output": "上下文压缩成功",
                }
            )
        except Exception:
            return

    def consume_auto_tool_traces(self) -> list[dict[str, str]]:
        rows = list(self._auto_tool_traces)
        self._auto_tool_traces = []
        return rows

    def _build_compact_session_memory_prompt(self, session_memory: str) -> str:
        recent_tail = self._render_recent_session_messages()
        return (
            "请将下面的会话记忆压缩为高密度摘要。\n"
            f"目标不超过约 {SESSION_MEMORY_COMPACT_TARGET_TOKENS} token。\n"
            "必须保留：用户偏好、进行中的任务、约束、已完成结论、未完成事项、关键路径与文件。\n"
            "输出要求：只输出压缩结果正文，不要解释，不要 markdown 标题。\n"
            "最后必须单独追加一行‘最新任务状态：...’。\n"
            "判断最新任务时，优先参考最近会话尾部，而不是旧摘要里较早的任务。\n"
            "如果最近用户请求是压缩上下文、compact_context、上下文压缩或整理记忆，最后一行必须写‘最新任务状态：上下文压缩已完成，目前无任务’。\n"
            "如果没有明确未完成任务，写‘最新任务状态：无明确未完成任务’。\n\n"
            f"最近会话尾部：\n{recent_tail}\n\n"
            f"原始会话记忆：\n{session_memory}"
        )

    async def _compact_session_memory(self, session_memory: str) -> str:
        prompt = self._build_compact_session_memory_prompt(session_memory)
        response = await self._compact_model(
            [
                {
                    "role": "system",
                    "name": "system",
                    "content": "你是上下文压缩助手，只输出压缩后的会话记忆。",
                },
                {
                    "role": "user",
                    "name": "user",
                    "content": prompt,
                },
            ]
        )
        return self._extract_assistant_text(response)

    def _render_recent_session_messages(self, rounds: int = 3) -> str:
        try:
            rows = self._memory_store.get_recent_session_messages(
                self._user_id,
                self._session_id,
                rounds=rounds,
            )
        except Exception:
            return "(无)"

        if not rows:
            return "(无)"

        lines: list[str] = []
        for row in rows:
            role = str(row.get("role", "")).strip() or "unknown"
            name = str(row.get("name", "")).strip()
            content = str(row.get("content", "")).strip()
            if name:
                lines.append(f"{role}/{name}: {content}")
            else:
                lines.append(f"{role}: {content}")
        return "\n".join(lines).strip() or "(无)"

    def _estimate_text_tokens(self, text: str) -> int:
        content = str(text or "")
        chinese_chars = len([c for c in content if "\u4e00" <= c <= "\u9fff"])
        other_chars = max(0, len(content) - chinese_chars)
        return max(1, chinese_chars + ((other_chars + 3) // 4))

    def _message_field(self, msg: Any, field: str) -> Any:
        if isinstance(msg, dict):
            return msg.get(field)
        return getattr(msg, field, None)

    def _extract_assistant_text(self, value: Any) -> str:
        content = getattr(value, "content", None)
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return ""

        chunks: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text_value = block.get("text", "")
                else:
                    text_value = block.get("content", "")
                if text_value:
                    chunks.append(str(text_value))
                continue

            text_attr = getattr(block, "text", "")
            if text_attr:
                chunks.append(str(text_attr))

        return "\n".join(chunks).strip()

    def _serialize_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): self._serialize_value(item) for key, item in value.items()
            }
        if isinstance(value, str):
            return value

        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            try:
                return self._serialize_value(model_dump())
            except Exception:
                pass

        dict_method = getattr(value, "dict", None)
        if callable(dict_method):
            try:
                return self._serialize_value(dict_method())
            except Exception:
                pass

        serialized: dict[str, Any] = {}
        for attr in ("role", "name", "content"):
            attr_value = getattr(value, attr, None)
            if attr_value is not None:
                serialized[attr] = self._serialize_value(attr_value)
        if serialized:
            return serialized
        return repr(value)


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
        self._pending_runtime_agents: dict[str, Any] = {}
        self._tool_trace_queues: dict[str, asyncio.Queue] = {}
        self._buffered_tool_trace_events: dict[str, list[dict[str, Any]]] = {}
        self._max_buffered_tool_trace_events = 200
        self._pending_approval_trace_ids: dict[tuple[str, str, str], str] = {}

    def subscribe_tool_traces(self, session_id: str) -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()
        self._tool_trace_queues[session_id] = queue
        try:
            while True:
                try:
                    event = asyncio.wait_for(queue.get(), timeout=30)
                    if event is None:
                        break
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'heartbeat'}, ensure_ascii=False)}\n\n"
        finally:
            self._tool_trace_queues.pop(session_id, None)

    def push_tool_trace_event(self, session_id: str, event: dict[str, Any]) -> None:
        session_text = str(session_id or "").strip()
        if not session_text:
            return

        payload = dict(event) if isinstance(event, dict) else {}
        queue = self._tool_trace_queues.get(session_text)
        if queue is not None:
            try:
                queue.put_nowait(payload)
                return
            except Exception:
                pass

        buffered = self._buffered_tool_trace_events.setdefault(
            session_text, [])
        buffered.append(payload)
        overflow = len(buffered) - self._max_buffered_tool_trace_events
        if overflow > 0:
            del buffered[:overflow]

    def consume_buffered_tool_trace_events(self, session_id: str) -> list[dict[str, Any]]:
        session_text = str(session_id or "").strip()
        if not session_text:
            return []
        rows = self._buffered_tool_trace_events.pop(session_text, [])
        return rows if isinstance(rows, list) else []

    def remember_pending_approval_trace_id(
        self,
        user_id: str,
        session_id: str,
        request_id: str,
        trace_id: str,
    ) -> None:
        user_text = str(user_id or "").strip()
        session_text = str(session_id or "").strip()
        request_text = str(request_id or "").strip()
        trace_text = str(trace_id or "").strip()
        if not user_text or not session_text or not request_text or not trace_text:
            return
        self._pending_approval_trace_ids[(
            user_text, session_text, request_text)] = trace_text

    def pop_pending_approval_trace_id(
        self,
        user_id: str,
        session_id: str,
        request_id: str,
    ) -> str | None:
        key = (str(user_id or "").strip(), str(
            session_id or "").strip(), str(request_id or "").strip())
        if not all(key):
            return None
        return self._pending_approval_trace_ids.pop(key, None)

    def _respond_with_optional_session_memory(
        self,
        agent: object,
        user_message: str,
        long_term_context: str,
        session_memory: str,
    ) -> AgentResponse | str:
        respond = getattr(agent, "respond")
        try:
            signature = inspect.signature(respond)
        except (TypeError, ValueError):
            signature = None

        if signature is not None:
            parameters = signature.parameters
            if "session_memory" in parameters:
                return respond(
                    user_message,
                    long_term_context,
                    session_memory=session_memory,
                )

        if session_memory.strip():
            merged_context = (
                f"{long_term_context}\n\n[SESSION_MEMORY]\n{session_memory.strip()}"
            )
        else:
            merged_context = long_term_context
        return respond(user_message, merged_context)

    def chat(
        self,
        user_id: str,
        session_id: str,
        message: str,
        llm_user_message: str | None = None,
        use_latest_session_memory: bool = True,
        push_event_func: callable | None = None,
    ) -> ChatResult:
        self._validate_ids(user_id, session_id)

        def push_event(event_type: str, data: dict[str, Any]) -> None:
            if push_event_func:
                try:
                    push_event_func(
                        {"type": event_type, "session_id": session_id, **data}
                    )
                except Exception:
                    pass

        push_event("status", {"status": "started", "message": "开始处理请求"})

        effective_llm_user_message = (
            llm_user_message.strip()
            if isinstance(llm_user_message, str) and llm_user_message.strip()
            else message
        )

        previous_session_state = self.memory_store.get_session_state(
            user_id, session_id
        )
        previous_session_memory = str(
            previous_session_state.get("session_memory", "")
        ).strip()

        self.memory_store.append_session_message(
            user_id, session_id, "user", message)
        self.memory_store.append_daily_message(
            user_id, "user", message, meta={"session_id": session_id}
        )

        context = self.memory_store.load_context(
            user_id,
            session_id,
            query=message,
            session_limit=0,
            session_token_budget=2800,
            daily_limit=0,
        )
        prompt_context = self.memory_store.build_prompt_context(context)
        long_term_context = self.memory_store.build_long_term_prompt_context(
            context)
        session_memory = (
            context.session_memory
            if use_latest_session_memory
            else previous_session_memory
        )

        runtime_agent: Any = None
        collected_thinking_texts: list[str] = []
        collected_assistant_texts: list[str] = []
        collected_ui_timeline_events: list[dict[str, Any]] = []

        def append_ui_timeline_event(event_type: str, payload: dict[str, Any]) -> None:
            event_name = str(event_type or "").strip()
            if not event_name:
                return
            safe_payload = dict(payload) if isinstance(payload, dict) else {}
            item: dict[str, Any] = {"type": event_name, **safe_payload}
            item["order"] = len(collected_ui_timeline_events) + 1
            collected_ui_timeline_events.append(item)

        if self._agent is None:
            runtime_agent = ReActMemoryAgent(
                memory_store=self.memory_store, user_id=user_id, session_id=session_id
            )

            def emit_runtime_thinking_texts() -> None:
                consume = getattr(
                    runtime_agent, "consume_auto_thinking_texts", None)
                if not callable(consume):
                    return
                try:
                    rows = consume()
                except Exception:
                    return
                if not isinstance(rows, list):
                    return
                for raw_text in rows:
                    text = str(raw_text or "").strip()
                    if not text:
                        continue
                    collected_thinking_texts.append(text)
                    append_ui_timeline_event(
                        "assistant_thinking",
                        {
                            "text": text,
                        },
                    )
                    push_event(
                        "assistant_thinking",
                        {
                            "text": text,
                        },
                    )

            def emit_runtime_assistant_texts() -> None:
                consume = getattr(
                    runtime_agent, "consume_auto_assistant_texts", None)
                if not callable(consume):
                    return
                try:
                    rows = consume()
                except Exception:
                    return
                if not isinstance(rows, list):
                    return
                for raw_text in rows:
                    text = str(raw_text or "").strip()
                    if not text:
                        continue
                    collected_assistant_texts.append(text)
                    append_ui_timeline_event(
                        "assistant_text",
                        {
                            "text": text,
                        },
                    )
                    push_event(
                        "assistant_text",
                        {
                            "text": text,
                        },
                    )

            push_event(
                "status", {"status": "agent_ready", "message": "Agent 准备就绪"})

            if hasattr(runtime_agent, "_record_tool_trace"):
                original_record_tool_trace = runtime_agent._record_tool_trace

                def wrapped_record_tool_trace(name, input_value, output_value):
                    trace_id = uuid4().hex[:12]
                    output_text = runtime_agent._stringify_tool_value(
                        output_value)
                    emit_runtime_thinking_texts()
                    emit_runtime_assistant_texts()
                    push_event(
                        "tool_calling",
                        {
                            "trace_id": trace_id,
                            "name": name,
                            "input": runtime_agent._stringify_tool_value(input_value),
                        },
                    )
                    original_record_tool_trace(name, input_value, output_value)
                    approval_marker = "__APPROVAL_REQUIRED__"
                    if isinstance(output_text, str) and output_text.startswith(approval_marker):
                        raw_payload = output_text[len(
                            approval_marker):].strip()
                        try:
                            parsed_payload = json.loads(raw_payload)
                        except Exception:
                            parsed_payload = None
                        if isinstance(parsed_payload, dict):
                            request_id = str(parsed_payload.get(
                                "request_id", "")).strip()
                            if request_id:
                                self.remember_pending_approval_trace_id(
                                    user_id,
                                    session_id,
                                    request_id,
                                    trace_id,
                                )
                    push_event(
                        "tool_called",
                        {
                            "trace_id": trace_id,
                            "name": name,
                            "output": output_text,
                        },
                    )
                    append_ui_timeline_event(
                        "tool",
                        {
                            "trace_id": trace_id,
                            "name": str(name or "").strip() or "tool",
                            "input": runtime_agent._stringify_tool_value(input_value),
                            "output": output_text,
                        },
                    )

                runtime_agent._record_tool_trace = wrapped_record_tool_trace

            agent_result = self._respond_with_optional_session_memory(
                runtime_agent,
                effective_llm_user_message,
                long_term_context,
                session_memory,
            )
        else:
            raw_result = self._respond_with_optional_session_memory(
                self._agent,
                effective_llm_user_message,
                long_term_context,
                session_memory,
            )
            if isinstance(raw_result, AgentResponse):
                agent_result = raw_result
            else:
                agent_result = AgentResponse(
                    text=str(raw_result),
                    tool_calls=[],
                    loaded_skills=[],
                    activity_traces=[],
                )

            if runtime_agent is not None:
                emit_runtime_thinking_texts()
                emit_runtime_assistant_texts()

        push_event(
            "status", {"status": "generating_reply", "message": "正在生成回复"})

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
        if collected_thinking_texts:
            assistant_meta["ui_thinking_texts"] = collected_thinking_texts
        if collected_assistant_texts:
            assistant_meta["ui_assistant_texts"] = collected_assistant_texts
        if collected_ui_timeline_events:
            assistant_meta["ui_timeline_events"] = collected_ui_timeline_events

        record_session_memory = approval_request is None and not (
            runtime_agent is not None
            and getattr(runtime_agent, "_agent", None) is not None
        )

        self.memory_store.append_session_message(
            user_id,
            session_id,
            "assistant",
            reply,
            meta=assistant_meta,
            record_session_memory=record_session_memory,
        )
        self.memory_store.append_daily_message(
            user_id,
            "assistant",
            reply,
            meta={**assistant_meta, "session_id": session_id},
        )

        if approval_request and str(approval_request.get("request_id", "")).strip():
            req_id = str(approval_request.get("request_id", "")).strip()
            if runtime_agent is not None:
                self._pending_runtime_agents[req_id] = runtime_agent
            self.memory_store.set_pending_react_state(
                user_id,
                session_id,
                req_id,
                {"user_message": message, "prompt_context": prompt_context},
            )

        if choice_request and str(choice_request.get("request_id", "")).strip():
            self.memory_store.create_pending_choice(
                user_id, session_id, choice_request)

        title = self.memory_store.get_conversation_title(user_id, session_id)
        if not title:
            if self._title_generator is None:
                generator = ConversationTitleGenerator()
                title = generator.generate_title(message, "")
            else:
                title = self._title_generator.generate_title(message, "")
        self.memory_store.set_conversation_title(user_id, session_id, title)

        push_event("status", {"status": "completed", "message": "处理完成"})

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
            user_id, session_id, req_id, output
        )
        self.memory_store.append_session_memory_tool_trace(
            user_id, session_id, "bash_command", command, output
        )
        state = self.memory_store.pop_pending_react_state(
            user_id, session_id, req_id)
        runtime_agent = self._pending_runtime_agents.pop(req_id, None)

        initial_tool_call = {"name": "bash_command",
                             "input": command, "output": output}

        if state is None and runtime_agent is None:
            reply = (
                f"已{'批准' if approved else '拒绝'}执行命令\n$ {command}\n\n{output}"
            )
            meta = {"tool_calls": [initial_tool_call]}
            self.memory_store.append_session_message(
                user_id, session_id, "assistant", reply, meta=meta
            )
            self.memory_store.append_daily_message(
                user_id, "assistant", reply, meta={**meta, "session_id": session_id}
            )
            title = (
                self.memory_store.get_conversation_title(user_id, session_id)
                or "新对话"
            )
            self.memory_store.set_conversation_title(
                user_id, session_id, title)
            return ChatResult(
                reply=reply, memory_hits=0, title=title, tool_calls=[initial_tool_call]
            )

        user_message = (
            str(state.get("user_message", "")).strip()
            if isinstance(state, dict)
            else ""
        )
        if not user_message:
            user_message = "继续当前任务"

        context = self.memory_store.load_context(
            user_id,
            session_id,
            query=user_message,
            session_limit=0,
            session_token_budget=2800,
            daily_limit=0,
        )
        prompt_context = self.memory_store.build_prompt_context(context)
        long_term_context = self.memory_store.build_long_term_prompt_context(
            context)
        session_memory = context.session_memory

        if self._agent is None:
            if runtime_agent is None:
                runtime_agent = ReActMemoryAgent(
                    memory_store=self.memory_store,
                    user_id=user_id,
                    session_id=session_id,
                )
            agent_result = self._respond_with_optional_session_memory(
                runtime_agent, user_message, long_term_context, session_memory
            )
        else:
            raw_result = self._respond_with_optional_session_memory(
                self._agent, user_message, long_term_context, session_memory
            )
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

        record_session_memory = approval_request is None and not (
            runtime_agent is not None
            and getattr(runtime_agent, "_agent", None) is not None
        )

        self.memory_store.append_session_message(
            user_id,
            session_id,
            "assistant",
            reply,
            meta=assistant_meta,
            record_session_memory=record_session_memory,
        )
        self.memory_store.append_daily_message(
            user_id,
            "assistant",
            reply,
            meta={**assistant_meta, "session_id": session_id},
        )

        if approval_request and str(approval_request.get("request_id", "")).strip():
            next_req_id = str(approval_request.get("request_id", "")).strip()
            if runtime_agent is not None:
                self._pending_runtime_agents[next_req_id] = runtime_agent
            self.memory_store.set_pending_react_state(
                user_id,
                session_id,
                next_req_id,
                {"user_message": user_message, "prompt_context": prompt_context},
            )

        if choice_request and str(choice_request.get("request_id", "")).strip():
            self.memory_store.create_pending_choice(
                user_id, session_id, choice_request)

        title = self.memory_store.get_conversation_title(user_id, session_id)
        if not title:
            if self._title_generator is None:
                title = ConversationTitleGenerator().generate_title(user_message, "")
            else:
                title = self._title_generator.generate_title(user_message, "")
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

    def create_conversation(self, user_id: str) -> str:
        self._validate_ids(user_id, "seed")
        return uuid4().hex[:12]

    def list_conversations(self, user_id: str) -> list[dict[str, str]]:
        self._validate_ids(user_id, "seed")
        return self.memory_store.list_conversations(user_id)

    def delete_conversation(self, user_id: str, session_id: str) -> bool:
        self._validate_ids(user_id, session_id)
        return self.memory_store.delete_conversation(user_id, session_id)

    def rename_conversation(self, user_id: str, session_id: str, title: str) -> str:
        self._validate_ids(user_id, session_id)
        title_text = title.strip()
        if not title_text:
            raise ValueError("title is required")
        self.memory_store.set_conversation_title(
            user_id, session_id, title_text)
        return self.memory_store.get_conversation_title(user_id, session_id) or "新对话"

    def get_session_messages(
        self, user_id: str, session_id: str
    ) -> list[dict[str, Any]]:
        self._validate_ids(user_id, session_id)
        rows = self.memory_store.get_session_messages(user_id, session_id)
        result: list[dict[str, Any]] = []
        for row in rows:
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            raw_meta = dict(meta) if isinstance(meta, dict) else {}
            raw_tool_calls = meta.get(
                "tool_calls") if isinstance(meta, dict) else []
            tool_calls: list[dict[str, str]] = []
            if isinstance(raw_tool_calls, list):
                for item in raw_tool_calls:
                    if isinstance(item, dict):
                        tool_calls.append(
                            {
                                "name": str(item.get("name", "")),
                                "input": str(item.get("input", "")),
                                "output": str(item.get("output", "")),
                            }
                        )

            raw_loaded_skills = (
                meta.get("loaded_skills") if isinstance(meta, dict) else []
            )
            loaded_skills: list[dict[str, str]] = []
            if isinstance(raw_loaded_skills, list):
                for item in raw_loaded_skills:
                    if isinstance(item, dict):
                        loaded_skills.append(
                            {
                                "name": str(item.get("name", "")),
                                "source": str(item.get("source", "")),
                                "dir": str(item.get("dir", "")),
                            }
                        )

            raw_activity_traces = (
                meta.get("activity_traces") if isinstance(meta, dict) else []
            )
            activity_traces: list[dict[str, str]] = []
            if isinstance(raw_activity_traces, list):
                for item in raw_activity_traces:
                    if isinstance(item, dict):
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
                    "message_id": int(row.get("message_id", 0))
                    if str(row.get("message_id", "")).isdigit()
                    else 0,
                    "meta": raw_meta,
                    "tool_calls": tool_calls,
                    "loaded_skills": loaded_skills,
                    "activity_traces": activity_traces,
                }
            )
        return result

    def _validate_ids(self, user_id: str, session_id: str) -> None:
        if not user_id.strip() or not session_id.strip():
            raise ValueError("user_id and session_id are required")

    def _filter_placeholder_tool_calls(
        self, tool_calls: list[dict[str, str]]
    ) -> list[dict[str, str]]:
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

    def _filter_placeholder_activity_traces(
        self, traces: list[dict[str, str]]
    ) -> list[dict[str, str]]:
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

    def _extract_approval_request(
        self, tool_calls: list[dict[str, str]]
    ) -> dict[str, Any] | None:
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

    def _extract_choice_request(
        self, tool_calls: list[dict[str, str]]
    ) -> dict[str, Any] | None:
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
