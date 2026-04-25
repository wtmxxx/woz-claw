from __future__ import annotations

import asyncio
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentscope.agent import ReActAgent
from agentscope.formatter import DeepSeekChatFormatter, OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import ToolResponse, Toolkit
import yaml

from wozclaw.config import load_llm_config, load_path_config
from wozclaw.memory_store import MemoryStore


SESSION_MEMORY_COMPACT_THRESHOLD_TOKENS = 30000
SESSION_MEMORY_COMPACT_TARGET_TOKENS = 8000


@dataclass
class AgentResponse:
    text: str
    tool_calls: list[dict[str, str]] = field(default_factory=list)
    loaded_skills: list[dict[str, str]] = field(default_factory=list)
    activity_traces: list[dict[str, str]] = field(default_factory=list)


class ApprovalInterrupt(Exception):
    """Raised to stop the current ReAct loop immediately when human approval is required."""


class ChoiceInterrupt(Exception):
    """Raised to stop the current ReAct loop immediately when human choice is required."""


class FallbackAgent:
    def respond(self, user_message: str, memory_context: str) -> str:
        # Keep fallback deterministic for local demo and tests.
        if "中文" in memory_context or "language" in memory_context.lower():
            return f"收到：{user_message}。我会继续用中文回答。"
        return f"收到：{user_message}。我已经记录到记忆中。"


class LLMDialogueRecorder:
    """Wraps OpenAIChatModel to record each LLM call to dialogue log."""

    def __init__(
        self,
        model: OpenAIChatModel,
        memory_store: MemoryStore,
        user_id: str,
        session_id: str,
        compact_model: OpenAIChatModel | None = None,
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
        self._auto_assistant_texts: list[str] = []
        self._auto_thinking_texts: list[str] = []

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
            thinking_text = self._extract_thinking_text(response)
            if thinking_text:
                self._auto_thinking_texts.append(thinking_text)
            assistant_text = self._extract_assistant_text(response)
            if assistant_text:
                self._auto_assistant_texts.append(assistant_text)
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
            pass  # Don't fail the agent if logging fails
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
            # Temporarily disable the third user message.
            # filtered.append(latest_user_msg)
            pass

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

    def consume_auto_assistant_texts(self) -> list[str]:
        rows = [str(item).strip()
                for item in self._auto_assistant_texts if str(item).strip()]
        self._auto_assistant_texts = []
        return rows

    def consume_auto_thinking_texts(self) -> list[str]:
        rows = [
            str(item).strip()
            for item in self._auto_thinking_texts
            if str(item).strip()
        ]
        self._auto_thinking_texts = []
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

    def _render_recent_session_messages(self, rounds: int = 6) -> str:
        try:
            messages = self._memory_store.get_recent_session_messages(
                self._user_id,
                self._session_id,
                rounds=max(1, int(rounds)),
            )
        except Exception:
            messages = []
        if not messages:
            return "(无最近会话消息)"

        lines: list[str] = []
        for msg in messages:
            role = str(msg.get("role", ""))
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                prefix = "user"
            elif role == "assistant":
                prefix = "assistant"
            else:
                prefix = role or "message"
            lines.append(f"{prefix}: {content}")
        if not lines:
            return "(无最近会话消息)"
        return "\n".join(lines)

    def _estimate_text_tokens(self, text: str) -> int:
        content = str(text or "")
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", content))
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

    def _extract_thinking_text(self, value: Any) -> str:
        content = getattr(value, "content", None)
        if not isinstance(content, list):
            return ""

        chunks: list[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = str(block.get("type", "")).strip().lower()
                if block_type in {"thinking", "reasoning"}:
                    text_value = block.get("thinking") or block.get(
                        "text") or block.get("content", "")
                    if text_value:
                        chunks.append(str(text_value))
                continue

            block_type = str(getattr(block, "type", "")).strip().lower()
            if block_type in {"thinking", "reasoning"}:
                thinking_text = getattr(block, "thinking", "") or getattr(
                    block, "text", "") or getattr(block, "content", "")
                if thinking_text:
                    chunks.append(str(thinking_text))

        return "\n".join(chunks).strip()

    def _serialize_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, list):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._serialize_value(item) for key, item in value.items()}
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


class ReActMemoryAgent:
    def __init__(self, memory_store: MemoryStore, user_id: str, session_id: str) -> None:
        self.memory_store = memory_store
        self.user_id = user_id
        self.session_id = session_id
        self._approval_interrupt_requested = False
        self._approval_interrupt_message = "检测到需要人工审批，已暂停执行并等待审批结果。"
        self._choice_interrupt_requested = False
        self._choice_interrupt_message = "检测到需要做出选择，已暂停执行并等待用户选择。"
        self._bash_work_dir = self._project_root_dir().resolve()
        self._active_session_memory = ""
        self._active_memory_context = ""

        llm_config = load_llm_config()
        if not llm_config.api_key:
            self._agent = None
            self._fallback = FallbackAgent()
            self._loaded_skills: list[dict[str, str]] = []
            return
        self._active_tool_traces: list[dict[str, str]] = []
        self._active_activity_traces: list[dict[str, str]] = []
        self._loaded_skills: list[dict[str, str]] = []
        self._compact_model: OpenAIChatModel | None = None
        self._dialogue_recorder: LLMDialogueRecorder | None = None
        model = self._build_chat_model(
            api_key=llm_config.api_key,
            model_name=llm_config.model,
            base_url=llm_config.base_url,
            temperature=0.2,
            compact_model_name=llm_config.compact_model,
        )
        formatter = self._build_formatter(
            model_name=llm_config.model,
            base_url=llm_config.base_url,
        )
        toolkit = self._build_toolkit()
        self._agent = ReActAgent(
            name="memory-assistant",
            sys_prompt=self.build_system_prompt(""),
            model=model,
            formatter=formatter,
            toolkit=toolkit,
            enable_rewrite_query=False,
            print_hint_msg=False,
            max_iters=100,
        )
        self._fallback = FallbackAgent()

    def respond(
        self,
        user_message: str,
        memory_context: str,
        session_memory: str = "",
    ) -> AgentResponse:
        if self._agent is None:
            return AgentResponse(
                text="LLM 未配置，请在 config/llm.yaml 中设置 api_key 后重试。",
                tool_calls=[],
                loaded_skills=[],
                activity_traces=[],
            )

        self._active_memory_context = memory_context
        self._active_session_memory = session_memory.strip()
        prompt = self.build_system_prompt(memory_context)

        self._active_tool_traces = []
        self._active_activity_traces = [
            {
                "type": "skill",
                "name": item.get("name", ""),
                "source": item.get("source", ""),
                "dir": item.get("dir", ""),
                "input": "",
                "output": "loaded",
            }
            for item in self._loaded_skills
        ]
        self._set_runtime_prompt(prompt)
        self._discard_stale_auto_tool_traces()
        self._discard_stale_auto_assistant_texts()
        self._discard_stale_auto_thinking_texts()
        self._approval_interrupt_requested = False
        self._approval_interrupt_message = "检测到需要人工审批，已暂停执行并等待审批结果。"
        self._choice_interrupt_requested = False
        self._choice_interrupt_message = "检测到需要做出选择，已暂停执行并等待用户选择。"

        prompt_messages = []
        session_memory_text = session_memory.strip()
        if session_memory_text:
            prompt_messages.append(
                Msg(name="session_memory", content=session_memory_text, role="user"),
            )
        prompt_messages.append(
            Msg(name="user", content=user_message, role="user"))

        try:
            reply_msg = self._run_async_with_interrupt(
                self._agent.reply(prompt_messages)
            )
        except ApprovalInterrupt as exc:
            self._merge_auto_tool_traces_into_runtime_traces()
            return AgentResponse(
                text=str(exc) or "检测到需要人工审批，已暂停执行。",
                tool_calls=list(self._active_tool_traces),
                loaded_skills=list(self._loaded_skills),
                activity_traces=list(self._active_activity_traces),
            )
        except ChoiceInterrupt as exc:
            self._merge_auto_tool_traces_into_runtime_traces()
            return AgentResponse(
                text=str(exc) or "检测到需要做出选择，已暂停执行。",
                tool_calls=list(self._active_tool_traces),
                loaded_skills=list(self._loaded_skills),
                activity_traces=list(self._active_activity_traces),
            )
        except Exception as exc:
            self._merge_auto_tool_traces_into_runtime_traces()
            return AgentResponse(
                text=f"LLM 调用失败：{type(exc).__name__}: {exc}",
                tool_calls=list(self._active_tool_traces),
                loaded_skills=list(self._loaded_skills),
                activity_traces=list(self._active_activity_traces),
            )

        reply_text = self._msg_to_text(reply_msg)
        if not reply_text:
            reply_text = "LLM 返回空内容，请重试。"

        self._merge_auto_tool_traces_into_runtime_traces()

        return AgentResponse(
            text=reply_text,
            tool_calls=list(self._active_tool_traces),
            loaded_skills=list(self._loaded_skills),
            activity_traces=list(self._active_activity_traces),
        )

    def _build_chat_model(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
        temperature: float,
        compact_model_name: str = "",
    ) -> Any:
        client_kwargs: dict[str, Any] = {}
        if base_url.strip():
            client_kwargs["base_url"] = base_url.strip()
        model = OpenAIChatModel(
            model_name=model_name,
            api_key=api_key,
            stream=False,
            client_kwargs=client_kwargs or None,
            generate_kwargs={"temperature": temperature},
        )
        compact_name = compact_model_name.strip() or model_name
        try:
            self._compact_model = OpenAIChatModel(
                model_name=compact_name,
                api_key=api_key,
                stream=False,
                client_kwargs=client_kwargs or None,
                generate_kwargs={"temperature": 0},
            )
        except Exception:
            self._compact_model = None
        recorder = LLMDialogueRecorder(
            model,
            self.memory_store,
            self.user_id,
            self.session_id,
            compact_model=self._compact_model,
            compact_threshold_tokens=SESSION_MEMORY_COMPACT_THRESHOLD_TOKENS,
        )
        self._dialogue_recorder = recorder
        return recorder

    def _build_formatter(self, model_name: str, base_url: str) -> Any:
        model_text = str(model_name or "").strip().lower()
        base_url_text = str(base_url or "").strip().lower()
        if "deepseek" in model_text or "deepseek" in base_url_text:
            return DeepSeekChatFormatter()
        return OpenAIChatFormatter()

    def _discard_stale_auto_tool_traces(self) -> None:
        recorder = self._dialogue_recorder
        if recorder is None:
            return
        consume = getattr(recorder, "consume_auto_tool_traces", None)
        if not callable(consume):
            return
        try:
            consume()
        except Exception:
            return

    def _discard_stale_auto_assistant_texts(self) -> None:
        recorder = self._dialogue_recorder
        if recorder is None:
            return
        consume = getattr(recorder, "consume_auto_assistant_texts", None)
        if not callable(consume):
            return
        try:
            consume()
        except Exception:
            return

    def _discard_stale_auto_thinking_texts(self) -> None:
        recorder = self._dialogue_recorder
        if recorder is None:
            return
        consume = getattr(recorder, "consume_auto_thinking_texts", None)
        if not callable(consume):
            return
        try:
            consume()
        except Exception:
            return

    def consume_auto_assistant_texts(self) -> list[str]:
        recorder = self._dialogue_recorder
        if recorder is None:
            return []
        consume = getattr(recorder, "consume_auto_assistant_texts", None)
        if not callable(consume):
            return []
        try:
            rows = consume()
        except Exception:
            return []
        if not isinstance(rows, list):
            return []
        return [str(item).strip() for item in rows if str(item).strip()]

    def consume_auto_thinking_texts(self) -> list[str]:
        recorder = self._dialogue_recorder
        if recorder is None:
            return []
        consume = getattr(recorder, "consume_auto_thinking_texts", None)
        if not callable(consume):
            return []
        try:
            rows = consume()
        except Exception:
            return []
        if not isinstance(rows, list):
            return []
        return [str(item).strip() for item in rows if str(item).strip()]

    def _merge_auto_tool_traces_into_runtime_traces(self) -> None:
        recorder = self._dialogue_recorder
        if recorder is None:
            return
        consume = getattr(recorder, "consume_auto_tool_traces", None)
        if not callable(consume):
            return

        try:
            auto_rows = consume()
        except Exception:
            return

        if not isinstance(auto_rows, list):
            return

        for raw in auto_rows:
            if not isinstance(raw, dict):
                continue
            item = {
                "name": str(raw.get("name", "")).strip() or "tool",
                "input": self._stringify_tool_value(raw.get("input", "")),
                "output": self._stringify_tool_value(raw.get("output", "")),
            }
            self._active_tool_traces.append(item)
            self._active_activity_traces.append(
                {
                    "type": "tool",
                    "name": item["name"],
                    "source": "",
                    "dir": "",
                    "input": item["input"],
                    "output": item["output"],
                }
            )

    def _build_toolkit(self) -> Toolkit:
        toolkit = Toolkit(
            agent_skill_instruction=(
                "<SKILLS>\n"
                "The agent skills are a collection of folders of instructions, scripts, and resources. "
                "If you want to use a skill, you MUST read its SKILL.md file carefully first. "
                "Some skills include multiple files (for example references/, assets/, or scripts/) in the same directory, "
                "and you may read those files when needed.\n"
                "</SKILLS>"
            ),
            agent_skill_template=(
                "<SKILL>\n"
                "<NAME>{name}</NAME>\n"
                "<DESCRIPTION>{description}</DESCRIPTION>\n"
                "<USAGE>Check \"{dir}/SKILL.md\" for how to use this skill.</USAGE>\n"
                "</SKILL>"
            ),
        )

        def remember_note(note: str, tags: str = "preference,fact") -> ToolResponse:
            """Overwrite memory.md with the latest full-memory text. The input must be complete, not incremental."""
            input_payload = {"note": note, "tags": tags}
            try:
                tag_list = [item.strip()
                            for item in tags.split(",") if item.strip()]
                self.memory_store.remember_long_term(
                    self.user_id, note, tags=tag_list)
                output = "stored"
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace("remember_note", input_payload, output)
            return self._to_tool_response(output)

        def get_recent_session(rounds: int = 3) -> ToolResponse:
            """Fetch the latest session history by rounds (1 round ~= user+assistant messages)."""
            input_payload = {"rounds": rounds}
            try:
                rows = self.memory_store.get_recent_session_messages(
                    self.user_id, self.session_id, rounds=max(1, rounds))
                if not rows:
                    output = "no session history"
                else:
                    output = "\n".join(
                        f"#{row.get('message_id', '')} {row.get('ts', '')} | {row.get('role', '')}: {row.get('content', '')}"
                        for row in rows
                    )
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace(
                "get_recent_session", input_payload, output)
            return self._to_tool_response(output)

        def search_session(keyword: str, limit: int = 20) -> ToolResponse:
            """Search session history by keyword and return message_id anchors for follow-up window lookups."""
            input_payload = {"keyword": keyword, "limit": limit}
            try:
                rows = self.memory_store.search_session_messages(
                    self.user_id,
                    self.session_id,
                    keyword=keyword,
                    limit=max(1, min(limit, 100)),
                )
                if not rows:
                    output = "no matched session messages"
                else:
                    output = "\n".join(
                        f"#{row.get('message_id', '')} {row.get('ts', '')} | {row.get('role', '')}: {row.get('content', '')}"
                        for row in rows
                    )
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace("search_session", input_payload, output)
            return self._to_tool_response(output)

        def get_session_window(message_id: int, before: int = 0, after: int = 0) -> ToolResponse:
            """Fetch session messages around a message_id anchor. Use after search_session to expand context."""
            input_payload = {"message_id": message_id,
                             "before": before, "after": after}
            try:
                rows = self.memory_store.get_session_messages_window(
                    self.user_id,
                    self.session_id,
                    message_id=message_id,
                    before=max(0, before),
                    after=max(0, after),
                )
                if not rows:
                    output = "no window messages"
                else:
                    output = "\n".join(
                        f"#{row.get('message_id', '')} {row.get('ts', '')} | {row.get('role', '')}: {row.get('content', '')}"
                        for row in rows
                    )
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace(
                "get_session_window", input_payload, output)
            return self._to_tool_response(output)

        def search_daily(keyword: str, day: str = "", limit: int = 50) -> ToolResponse:
            """Search daily history by keyword. Optional day=YYYY-MM-DD; empty/invalid day falls back to all days."""
            input_payload = {"keyword": keyword, "day": day, "limit": limit}
            try:
                rows = self.memory_store.search_daily_messages(
                    self.user_id,
                    keyword=keyword,
                    limit=max(1, min(limit, 200)),
                    day=day,
                )
                if not rows:
                    output = "no matched daily messages"
                else:
                    output = "\n".join(
                        f"#{row.get('message_id', '')} {row.get('ts', '')} | {row.get('role', '')}: {row.get('content', '')}"
                        for row in rows
                    )
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace("search_daily", input_payload, output)
            return self._to_tool_response(output)

        def list_dir(path: str = ".") -> ToolResponse:
            """List files and folders under workdir-relative path."""
            input_payload = {"path": path}
            try:
                output = self._list_directory(path)
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace("list_dir", input_payload, output)
            return self._to_tool_response(output)

        def file_exists(path: str) -> ToolResponse:
            """Check whether a file or directory exists under workdir."""
            input_payload = {"path": path}
            try:
                output = self._file_exists(path)
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace("file_exists", input_payload, output)
            return self._to_tool_response(output)

        def read_file(path: str, start_line: int = 1, end_line: int = 200) -> ToolResponse:
            """Read text file content under workdir by line range (inclusive)."""
            input_payload = {
                "path": path,
                "start_line": start_line,
                "end_line": end_line,
            }
            try:
                output = self._read_text_file(path, start_line, end_line)
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace("read_file", input_payload, output)
            return self._to_tool_response(output)

        def write_file(path: str, content: str) -> ToolResponse:
            """Write full UTF-8 text content to a file under workdir."""
            input_payload = {"path": path, "content": content}
            try:
                output = self._write_text_file(path, content)
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace("write_file", input_payload, output)
            return self._to_tool_response(output)

        def append_file(path: str, content: str) -> ToolResponse:
            """Append UTF-8 text content to a file under workdir."""
            input_payload = {"path": path, "content": content}
            try:
                output = self._append_text_file(path, content)
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace("append_file", input_payload, output)
            return self._to_tool_response(output)

        def delete_file(path: str) -> ToolResponse:
            """Delete a file under workdir or .wozclaw."""
            input_payload = {"path": path}
            try:
                output = self._delete_file(path)
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace("delete_file", input_payload, output)
            return self._to_tool_response(output)

        def patch_file(path: str, patch_content: str) -> ToolResponse:
            """Apply unified diff patch to a file. patch_content should be in unified diff format.
            Example patch:
            --- a/file.txt
            +++ b/file.txt
            @@ -1,3 +1,3 @@
             line1
            -line2 old
            +line2 new
             line3
            """
            input_payload = {"path": path, "patch_content": patch_content}
            try:
                output = self._patch_text_file(path, patch_content)
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace("patch_file", input_payload, output)
            return self._to_tool_response(output)

        def search_file(path: str, pattern: str, limit: int = 50) -> ToolResponse:
            """Search for lines matching a pattern in a file (supports regex). Returns matching lines with line numbers."""
            input_payload = {"path": path, "pattern": pattern, "limit": limit}
            try:
                output = self._search_in_file(path, pattern, limit=limit)
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace("search_file", input_payload, output)
            return self._to_tool_response(output)

        def copy_file(src: str, dst: str) -> ToolResponse:
            """Copy a file from src to dst. Both paths must be readable (src) and writable (dst) locations."""
            input_payload = {"src": src, "dst": dst}
            try:
                output = self._copy_file(src, dst)
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace("copy_file", input_payload, output)
            return self._to_tool_response(output)

        def move_file(src: str, dst: str) -> ToolResponse:
            """Move or rename a file from src to dst. Both must be within allowed write directories."""
            input_payload = {"src": src, "dst": dst}
            try:
                output = self._move_file(src, dst)
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace("move_file", input_payload, output)
            return self._to_tool_response(output)

        def replace_file_lines(path: str, start_line: int, end_line: int, content: str) -> ToolResponse:
            """Replace file content in the specified line range (1-based, inclusive).

            Examples:
            - replace_file_lines("script.py", 10, 15, "new line 1\\nnew line 2")
              Replaces lines 10-15 with the new content
            - replace_file_lines("config.py", 5, 5, "DEBUG = False")
              Replaces only line 5
            - replace_file_lines("data.txt", 1, 100, "new content")
              Replaces first 100 lines with new content
            """
            input_payload = {"path": path, "start_line": start_line,
                             "end_line": end_line, "content": content}
            try:
                output = self._replace_file_lines(
                    path, start_line, end_line, content)
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace(
                "replace_file_lines", input_payload, output)
            return self._to_tool_response(output)

        def get_daily_window(message_id: int, before: int = 0, after: int = 0, day: str = "") -> ToolResponse:
            """Fetch daily messages around a message_id anchor. Use after search_daily to expand context."""
            input_payload = {
                "message_id": message_id,
                "before": before,
                "after": after,
                "day": day,
            }
            try:
                rows = self.memory_store.get_daily_messages_window(
                    self.user_id,
                    message_id=message_id,
                    before=max(0, before),
                    after=max(0, after),
                    day=day,
                )
                if not rows:
                    output = "no window messages"
                else:
                    output = "\n".join(
                        f"#{row.get('message_id', '')} {row.get('ts', '')} | {row.get('role', '')}: {row.get('content', '')}"
                        for row in rows
                    )
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace("get_daily_window", input_payload, output)
            return self._to_tool_response(output)

        def bash_command(command: str, background: bool = False) -> ToolResponse:
            """Run a policy-guarded bash command in the tracked current working directory."""
            input_payload = {"command": command, "background": background}
            try:
                output = self._run_bash_command(command, background=background)
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace(
                "bash_command", input_payload, output)
            if isinstance(output, str) and output.startswith("__APPROVAL_REQUIRED__"):
                self._approval_interrupt_requested = True
                self._approval_interrupt_message = "检测到需要人工审批，已暂停执行并等待审批结果。"
                raise ApprovalInterrupt("检测到需要人工审批，已暂停执行并等待审批结果。")
            return self._to_tool_response(output)

        def ask_human_choice(question: str, options: str = "", allow_custom: bool = True) -> ToolResponse:
            """Create a human-choice request when uncertain. options can be comma-separated or newline-separated."""
            input_payload = {
                "question": question,
                "options": options,
                "allow_custom": allow_custom,
            }
            try:
                normalized = [
                    item.strip()
                    for item in re.split(r"[\n,]", options)
                    if item.strip()
                ]
                choice = self.memory_store.create_pending_choice(
                    self.user_id,
                    self.session_id,
                    {
                        "type": "choice",
                        "question": question.strip(),
                        "options": normalized,
                        "allow_custom": bool(allow_custom),
                    },
                )
                payload = {
                    "request_id": choice.get("request_id", ""),
                    "question": question.strip(),
                    "options": normalized,
                    "allow_custom": bool(allow_custom),
                }
                output = "__CHOICE_REQUIRED__" + \
                    json.dumps(payload, ensure_ascii=False)
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace("ask_human_choice", input_payload, output)
            if isinstance(output, str) and output.startswith("__CHOICE_REQUIRED__"):
                self._choice_interrupt_requested = True
                self._choice_interrupt_message = "检测到需要做出选择，已暂停执行并等待用户选择。"
                raise ChoiceInterrupt("检测到需要做出选择，已暂停执行并等待用户选择。")
            return self._to_tool_response(output)

        def compact_context() -> ToolResponse:
            """Compact current session memory automatically and return success text."""
            input_payload: dict[str, Any] = {}
            try:
                self._compact_session_memory_now(force=True)
                output = "上下文压缩成功"
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace("compact_context", input_payload, output)
            return self._to_tool_response(output)

        toolkit.register_tool_function(remember_note)
        toolkit.register_tool_function(get_recent_session)
        toolkit.register_tool_function(search_session)
        toolkit.register_tool_function(get_session_window)
        toolkit.register_tool_function(search_daily)
        toolkit.register_tool_function(get_daily_window)
        toolkit.register_tool_function(list_dir)
        toolkit.register_tool_function(file_exists)
        toolkit.register_tool_function(read_file)
        toolkit.register_tool_function(write_file)
        toolkit.register_tool_function(append_file)
        toolkit.register_tool_function(delete_file)
        toolkit.register_tool_function(patch_file)
        toolkit.register_tool_function(search_file)
        toolkit.register_tool_function(copy_file)
        toolkit.register_tool_function(move_file)
        toolkit.register_tool_function(replace_file_lines)
        toolkit.register_tool_function(bash_command)
        toolkit.register_tool_function(ask_human_choice)
        toolkit.register_tool_function(compact_context)
        self._register_user_skills(toolkit)
        return toolkit

    def _validate_bash_command(self, command: str) -> tuple[bool, str]:
        text = command.strip()
        if not text:
            return False, "empty command"
        return True, ""

    def _expand_bash_aliases(self, command: str) -> str:
        return command

    def _root_work_dir(self, as_bash: bool = False) -> Path | str:
        configured = self._configured_work_dir()
        root_path: Path
        if configured is not None:
            root_path = configured
        elif self._bash_work_dir.exists() and self._bash_work_dir.is_dir():
            root_path = self._bash_work_dir
        else:
            self._bash_work_dir = self._project_root_dir().resolve()
            root_path = self._bash_work_dir

        if as_bash:
            return self._to_bash_path(root_path)
        return root_path

    def _extract_command_paths(self, command_name: str, args: list[str]) -> list[str]:
        if command_name in {"echo", "pwd"}:
            return []
        return [arg for arg in args if arg and not arg.startswith("-")]

    def _validate_bash_path(self, raw_path: str) -> tuple[bool, str]:
        _ = raw_path
        return True, ""

    def _allowed_bash_dirs(self) -> list[Path]:
        root_work_dir = self._root_work_dir()
        root_work_dir.mkdir(parents=True, exist_ok=True)
        return [root_work_dir.resolve()]

    def _configured_work_dir(self) -> Path | None:
        project_root = self._project_root_dir()
        config = load_path_config(project_root / "config" / "path.yaml")
        if not config.workdir:
            return None

        raw_path = config.workdir.strip()
        if not raw_path:
            return None

        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        return candidate

    def _configured_wozclaw_dir(self) -> Path | None:
        project_root = self._project_root_dir()
        config = load_path_config(project_root / "config" / "path.yaml")
        raw_path = config.wozclaw_dir.strip()
        if not raw_path:
            return None

        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        return candidate

    def _path_config_file(self) -> Path:
        return (self._project_root_dir() / "config" / "path.yaml").resolve()

    def _run_bash_command(self, command: str, background: bool = False) -> str:
        return self._run_bash_command_internal(
            command,
            skip_approval=False,
            background=background,
        )

    def run_bash_command_after_approval(self, command: str, background: bool = False) -> str:
        """Execute a previously approved command without re-entering approval gate."""
        return self._run_bash_command_internal(
            command,
            skip_approval=True,
            background=background,
        )

    def _run_bash_command_internal(
        self,
        command: str,
        skip_approval: bool = False,
        background: bool = False,
    ) -> str:
        ok, reason = self._validate_bash_command(command)
        if not ok:
            return (
                "error: unsafe bash command "
                f"({reason}). Avoid '..' in paths."
            )

        if not skip_approval:
            decision = self._evaluate_bash_policy(command)
            action = str(decision.get("action", "allow"))
            if action == "deny":
                return f"error: command denied by policy ({decision.get('reason', 'blocked')})"
            if action == "ask_human":
                approval = self.memory_store.create_pending_approval(
                    self.user_id,
                    self.session_id,
                    {
                        "type": "bash_command",
                        "command": command,
                        "background": bool(background),
                        "operation": str(decision.get("operation", "unknown")),
                        "reason": str(decision.get("reason", "requires human approval")),
                    },
                )
                payload = {
                    "request_id": approval.get("request_id", ""),
                    "command": command,
                    "background": bool(background),
                    "operation": str(decision.get("operation", "unknown")),
                    "reason": str(decision.get("reason", "requires human approval")),
                }
                return "__APPROVAL_REQUIRED__" + json.dumps(payload, ensure_ascii=False)

        work_dir = self._root_work_dir()
        work_dir.mkdir(parents=True, exist_ok=True)

        expanded_cmd = self._expand_bash_aliases(command)
        work_dir_str = str(self._root_work_dir(as_bash=False))
        base_command = f'cd "{work_dir_str}" && {expanded_cmd}'

        if background:
            cmd = ["bash", "-lc", base_command]
            try:
                process = subprocess.Popen(
                    cmd,
                    cwd=work_dir,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    shell=False,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    start_new_session=True,
                )
            except Exception as exc:
                return f"error: {exc}"

            payload = {
                "pid": getattr(process, "pid", 0),
                "command": command,
                "background": True,
            }
            return "__BACKGROUND_STARTED__" + json.dumps(payload, ensure_ascii=False)

        full_command = (
            f"{base_command}; "
            '__wozclaw_status=$?; '
            'printf "\n__WOZCLAW_CWD__%s\n" "$(pwd -P)"; '
            'exit $__wozclaw_status'
        )

        cmd = ["bash", "-lc", full_command]
        timeout_seconds = 30
        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                shell=False,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return "timeout"
        stdout = self._strip_terminal_control_sequences(
            result.stdout or "").strip()
        stderr = self._strip_terminal_control_sequences(
            result.stderr or "").strip()
        clean_stdout, cwd_marker = self._extract_cwd_marker(stdout)
        if cwd_marker:
            try:
                marker_path = Path(cwd_marker)
                if marker_path.exists() and marker_path.is_dir():
                    self._bash_work_dir = marker_path.resolve()
            except Exception:
                pass

        if result.returncode != 0:
            error_text = stderr or clean_stdout or f"command failed with exit code {result.returncode}"
            return f"error: {error_text}"

        output = clean_stdout[:100000] if clean_stdout else "ok"
        return output

    def _strip_terminal_control_sequences(self, text: str) -> str:
        if not text:
            return ""

        cleaned = str(text)
        # Strip CSI sequences such as clear-screen and cursor movement.
        cleaned = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", cleaned)
        # Strip OSC sequences.
        cleaned = re.sub(r"\x1b\][^\x07]*(?:\x07|\x1b\\)", "", cleaned)
        # Strip remaining single-character ESC sequences.
        cleaned = re.sub(r"\x1b[@-_]", "", cleaned)
        # Keep printable chars plus common whitespace used in command outputs.
        cleaned = "".join(
            ch for ch in cleaned if ch in "\n\r\t" or ord(ch) >= 32)
        return cleaned

    def _extract_cwd_marker(self, stdout: str) -> tuple[str, str]:
        marker = "__WOZCLAW_CWD__"
        lines = stdout.splitlines()
        if not lines:
            return stdout, ""

        for index in range(len(lines) - 1, -1, -1):
            line = lines[index]
            if line.startswith(marker):
                cwd = line[len(marker):].strip()
                kept = lines[:index] + lines[index + 1:]
                return "\n".join(kept).strip(), cwd
        return stdout, ""

    def _evaluate_bash_policy(self, command: str) -> dict[str, str]:
        policy = self.memory_store.get_command_policy(self.user_id)
        if not bool(policy.get("enabled", True)):
            return {"action": "allow", "operation": "read", "reason": "policy disabled"}

        command_name = self._extract_command_name(command)
        operation = self._classify_command_operation(command)
        allowlist = {
            str(item).strip().lower()
            for item in policy.get("command_allowlist", [])
            if str(item).strip()
        }
        blocklist = {
            str(item).strip().lower()
            for item in policy.get("command_blocklist", [])
            if str(item).strip()
        }

        if command_name and command_name in blocklist:
            return {"action": "deny", "operation": operation, "reason": f"{command_name} in blocklist"}

        if allowlist and command_name and command_name not in allowlist:
            return {
                "action": "ask_human",
                "operation": operation,
                "reason": f"{command_name} not in allowlist",
            }

        allowed_paths = policy.get("allowed_paths") if isinstance(
            policy.get("allowed_paths"), list) else []
        if operation in {"read", "write", "delete"} and self._paths_are_allowed(command, allowed_paths):
            return {
                "action": "allow",
                "operation": operation,
                "reason": "command paths in allowed_paths",
            }

        operations = policy.get("operations") if isinstance(
            policy.get("operations"), dict) else {}
        configured = str(operations.get(operation, policy.get(
            "default_action", "ask_human"))).strip().lower()
        if configured not in {"allow", "ask_human", "deny"}:
            configured = "ask_human"
        return {
            "action": configured,
            "operation": operation,
            "reason": f"operation {operation} policy",
        }

    def _extract_command_name(self, command: str) -> str:
        text = command.strip()
        if not text:
            return ""
        match = re.match(r"^[A-Za-z0-9_.-]+", text)
        if not match:
            return ""
        return match.group(0).lower()

    def _classify_command_operation(self, command: str) -> str:
        text = command.strip().lower()
        if not text:
            return "read"

        command_name = self._extract_command_name(command)
        if command_name in {
            "python", "python3", "bash", "sh", "zsh", "node", "npm", "pnpm", "yarn", "uv", "uvx",
        }:
            return "exec"

        delete_patterns = [r"\brm\b", r"\b-delete\b"]
        for pattern in delete_patterns:
            if re.search(pattern, text):
                return "delete"

        write_patterns = [
            r">", r"\btee\b", r"\bsed\s+-i\b", r"\bawk\b.*\b-i\b", r"\bperl\b.*\b-i\b",
            r"\bmv\b", r"\bcp\b", r"\btouch\b", r"\bmkdir\b",
        ]
        for pattern in write_patterns:
            if re.search(pattern, text):
                return "write"

        return "read"

    def _paths_are_allowed(self, command: str, allowed_paths: list[Any]) -> bool:
        normalized_allow = [
            self._normalize_policy_path(str(item))
            for item in allowed_paths
            if str(item).strip()
        ]
        allow_dirs = [item for item in normalized_allow if item is not None]
        if not allow_dirs:
            return False

        command_paths = self._extract_command_paths_for_policy(command)
        if not command_paths:
            return False

        for target in command_paths:
            if not any(self._is_path_under(target, base) for base in allow_dirs):
                return False
        return True

    def _extract_command_paths_for_policy(self, command: str) -> list[Path]:
        text = command.strip()
        if not text:
            return []

        tokens = re.findall(r'"[^\"]+"|\'[^\']+\'|[^\s]+', text)
        result: list[Path] = []
        for token in tokens[1:]:
            value = token.strip().strip('"\'')
            if not value or value.startswith("-"):
                continue
            if any(op in value for op in ["|", "&&", ";"]):
                continue
            if value in {">", ">>", "<", "<<"}:
                continue
            if "*" in value or "?" in value:
                continue
            if "/" not in value and "\\" not in value and not value.startswith("."):
                continue
            normalized = self._normalize_policy_path(value)
            if normalized is not None:
                result.append(normalized)
        return result

    def _normalize_policy_path(self, raw_path: str) -> Path | None:
        text = raw_path.strip()
        if not text:
            return None

        root_dir = self._root_work_dir().resolve()

        try:
            p = Path(text)
            if p.is_absolute():
                return p.resolve()

            return (root_dir / p).resolve()
        except Exception:
            return None

    def _is_path_under(self, target: Path, base: Path) -> bool:
        try:
            target.relative_to(base)
            return True
        except ValueError:
            return False

    def _rewrite_output_paths(self, output: str, work_dir: Path) -> str:
        _ = work_dir
        return output

    def _resolve_read_file_path(self, raw_path: str, allow_missing: bool = False) -> Path | None:
        """Resolve file path for read operations - no restrictions on path location."""
        text = str(raw_path or "").strip()
        if not text:
            text = "."

        try:
            candidate = Path(text)
            resolved = candidate.resolve(strict=False)
        except Exception:
            return None

        if not allow_missing and not resolved.exists():
            return None
        return resolved

    def _resolve_write_file_path(self, raw_path: str, allow_missing: bool = False) -> Path | None:
        """Resolve file path for write operations - restricted to .wozclaw and workdir."""
        text = str(raw_path or "").strip()
        if not text:
            text = "."

        work_root = self._root_work_dir().resolve()
        wozclaw_root = self._wozclaw_dir()
        if isinstance(wozclaw_root, str):
            wozclaw_root = Path(wozclaw_root)
        wozclaw_root = wozclaw_root.resolve()

        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = work_root / candidate

        try:
            resolved = candidate.resolve(strict=False)
        except Exception:
            return None

        # Allow paths under workdir or .wozclaw
        is_under_workdir = self._is_path_under(resolved, work_root)
        is_under_wozclaw = self._is_path_under(resolved, wozclaw_root)

        if not (is_under_workdir or is_under_wozclaw):
            return None

        if not allow_missing and not resolved.exists():
            return None
        return resolved

    def _resolve_file_tool_path(self, raw_path: str, allow_missing: bool = False) -> Path | None:
        """Deprecated: Use _resolve_read_file_path or _resolve_write_file_path instead."""
        return self._resolve_write_file_path(raw_path, allow_missing)

    def _list_directory(self, path: str = ".") -> str:
        target = self._resolve_read_file_path(path)
        if target is None:
            return "error: path does not exist"
        if not target.exists():
            return "error: path does not exist"
        if not target.is_dir():
            return "error: target is not a directory"

        rows: list[str] = []
        for child in sorted(target.iterdir(), key=lambda item: item.name.lower()):
            suffix = "/" if child.is_dir() else ""
            rows.append(f"{child.name}{suffix}")
            if len(rows) >= 500:
                break
        return "\n".join(rows) if rows else "(empty)"

    def _file_exists(self, path: str) -> str:
        target = self._resolve_read_file_path(path, allow_missing=True)
        if target is None:
            return "error: invalid path"
        if not target.exists():
            return "exists=false"
        kind = "dir" if target.is_dir() else "file"
        return f"exists=true type={kind}"

    def _read_text_file(self, path: str, start_line: int = 1, end_line: int = 200) -> str:
        target = self._resolve_read_file_path(path)
        if target is None:
            return "error: file not found"
        if not target.is_file():
            return "error: target is not a file"

        start = max(1, int(start_line))
        end = max(start, int(end_line))

        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if not lines:
            return ""
        chunk = lines[start - 1:end]
        if not chunk:
            return ""
        return "\n".join(chunk)[:100000]

    def _write_text_file(self, path: str, content: str) -> str:
        target = self._resolve_write_file_path(path, allow_missing=True)
        if target is None:
            return "error: path is outside allowed directories (.wozclaw or workdir)"
        if target.exists() and target.is_dir():
            return "error: target is a directory"

        target.parent.mkdir(parents=True, exist_ok=True)
        payload = str(content)
        target.write_text(payload, encoding="utf-8")
        return f"written {len(payload)} chars"

    def _append_text_file(self, path: str, content: str) -> str:
        target = self._resolve_write_file_path(path, allow_missing=True)
        if target is None:
            return "error: path is outside allowed directories (.wozclaw or workdir)"
        if target.exists() and target.is_dir():
            return "error: target is a directory"

        target.parent.mkdir(parents=True, exist_ok=True)
        payload = str(content)
        with target.open("a", encoding="utf-8") as file_obj:
            file_obj.write(payload)
        return f"appended {len(payload)} chars"

    def _delete_file(self, path: str) -> str:
        target = self._resolve_write_file_path(path)
        if target is None:
            return "error: path is outside allowed directories (.wozclaw or workdir)"
        if not target.exists():
            return "error: file does not exist"
        if not target.is_file():
            return "error: target is not a file"

        target.unlink()
        return "deleted"

    def _patch_text_file(self, path: str, patch_content: str) -> str:
        """Apply unified diff patch to a file. patch_content should be unified diff format."""
        import difflib
        import re as regex_module

        target = self._resolve_write_file_path(path)
        if target is None:
            return "error: path is outside allowed directories (.wozclaw or workdir)"
        if not target.exists():
            return "error: file does not exist"
        if not target.is_file():
            return "error: target is not a file"

        try:
            original_content = target.read_text(
                encoding="utf-8", errors="replace")
            original_lines = original_content.splitlines(keepends=True)

            # Parse the patch
            patch_lines = patch_content.splitlines(keepends=True)
            if not patch_lines:
                return "error: empty patch content"

            # Apply patch using simple line-based patching
            patched_lines = self._apply_unified_diff(
                original_lines, patch_lines)
            if patched_lines is None:
                return "error: patch application failed - unable to apply patch to file"

            patched_content = "".join(patched_lines)
            target.write_text(patched_content, encoding="utf-8")
            return "patch applied successfully"
        except Exception as exc:
            return f"error: {exc}"

    def _apply_unified_diff(self, original_lines: list[str], patch_lines: list[str]) -> list[str] | None:
        """Apply unified diff patch to original lines."""
        result = list(original_lines)
        current_line = 0
        i = 0

        while i < len(patch_lines):
            line = patch_lines[i]

            # Skip headers and other metadata
            if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
                # Parse hunk header like @@ -10,5 +10,6 @@
                if line.startswith("@@"):
                    match = re.search(
                        r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                    if match:
                        # Convert to 0-based index
                        current_line = int(match.group(3)) - 1
                i += 1
                continue

            if line.startswith("-"):
                # Remove line
                if current_line < len(result) and result[current_line].rstrip("\n") == line[1:].rstrip("\n"):
                    result.pop(current_line)
                else:
                    return None  # Patch doesn't match
            elif line.startswith("+"):
                # Add line
                result.insert(current_line, line[1:])
                current_line += 1
            elif line.startswith(" "):
                # Context line - just advance
                current_line += 1
            elif line.startswith("\\"):
                # No newline indicator - skip
                pass
            else:
                # Other characters - might be file header or other metadata
                pass

            i += 1

        return result

    def _search_in_file(self, path: str, pattern: str, limit: int = 50) -> str:
        """Search for lines matching a pattern (regex) in a file."""
        target = self._resolve_read_file_path(path)
        if target is None:
            return "error: file not found"
        if not target.is_file():
            return "error: target is not a file"

        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            regex = re.compile(pattern)
            matches = []
            for line_num, line in enumerate(lines, 1):
                if regex.search(line):
                    matches.append(f"{line_num}: {line}")
                    if len(matches) >= limit:
                        break
            if not matches:
                return "no matches found"
            return "\n".join(matches)
        except re.error as exc:
            return f"error: invalid regex pattern - {exc}"
        except Exception as exc:
            return f"error: {exc}"

    def _copy_file(self, src: str, dst: str) -> str:
        """Copy a file from src to dst."""
        src_target = self._resolve_read_file_path(src)
        if src_target is None:
            return "error: source file not found"
        if not src_target.is_file():
            return "error: source is not a file"

        dst_target = self._resolve_write_file_path(dst, allow_missing=True)
        if dst_target is None:
            return "error: destination path is outside allowed directories (.wozclaw or workdir)"
        if dst_target.exists() and dst_target.is_dir():
            return "error: destination is a directory"

        try:
            dst_target.parent.mkdir(parents=True, exist_ok=True)
            content = src_target.read_bytes()
            dst_target.write_bytes(content)
            return f"copied {len(content)} bytes"
        except Exception as exc:
            return f"error: {exc}"

    def _move_file(self, src: str, dst: str) -> str:
        """Move or rename a file from src to dst."""
        src_target = self._resolve_write_file_path(src)
        if src_target is None:
            return "error: source path is outside allowed directories (.wozclaw or workdir)"
        if not src_target.exists():
            return "error: source file does not exist"
        if not src_target.is_file():
            return "error: source is not a file"

        dst_target = self._resolve_write_file_path(dst, allow_missing=True)
        if dst_target is None:
            return "error: destination path is outside allowed directories (.wozclaw or workdir)"
        if dst_target.exists() and dst_target.is_dir():
            return "error: destination is a directory"

        try:
            dst_target.parent.mkdir(parents=True, exist_ok=True)
            src_target.rename(dst_target)
            return f"moved {src} to {dst}"
        except Exception as exc:
            return f"error: {exc}"

    def _replace_file_lines(self, path: str, start_line: int, end_line: int, content: str) -> str:
        """Replace file content in the specified line range.

        Args:
            path: File path
            start_line: Starting line number (1-based, inclusive)
            end_line: Ending line number (1-based, inclusive)
            content: New content to insert (can be multi-line)

        Returns:
            Status message
        """
        target = self._resolve_write_file_path(path)
        if target is None:
            return "error: path is outside allowed directories (.wozclaw or workdir)"
        if not target.exists():
            return "error: file does not exist"
        if not target.is_file():
            return "error: target is not a file"

        try:
            # Read original content
            original_text = target.read_text(
                encoding="utf-8", errors="replace")
            lines = original_text.splitlines(keepends=True)

            # Validate line numbers
            start = max(1, int(start_line))
            end = max(start, int(end_line))

            if start > len(lines) + 1:
                return f"error: start_line {start} exceeds file length ({len(lines)})"

            effective_end = min(end, len(lines))

            # Prepare new content lines
            new_lines = content.splitlines(keepends=True)
            if content and not content.endswith('\n'):
                # Ensure last line has newline if original content needs it
                if new_lines:
                    new_lines[-1] = new_lines[-1] + '\n'
                else:
                    new_lines = [content]

            # Build result: keep lines before, insert new content, keep lines after
            result_lines = lines[:start - 1] + \
                new_lines + lines[effective_end:]

            # Join and write back
            result_text = ''.join(result_lines)
            target.write_text(result_text, encoding="utf-8")

            replaced_count = max(0, effective_end - start + 1)
            return f"replaced lines {start}-{effective_end} ({replaced_count} lines)"
        except Exception as exc:
            return f"error: {exc}"

    def _register_user_skills(self, toolkit: Toolkit) -> None:
        self._loaded_skills = []
        for item in self._resolve_enabled_skills():
            skill_dir = Path(item.get("dir", ""))
            if not str(skill_dir):
                continue
            try:
                toolkit.register_agent_skill(str(skill_dir))
                # Keep runtime registration path unchanged while rendering
                self._loaded_skills.append(
                    {
                        "name": item.get("name", ""),
                        "source": item.get("source", ""),
                        "dir": str(skill_dir),
                    }
                )
            except Exception:
                continue

    def _resolve_enabled_skill_dirs(self) -> list[Path]:
        return [Path(item["dir"]) for item in self._resolve_enabled_skills() if item.get("dir")]

    def _resolve_enabled_skills(self) -> list[dict[str, str]]:
        global_entries = self._load_skill_entries(
            self._global_skills_config_path(),
            scope="global",
        )
        user_entries = self._load_skill_entries(
            self._user_skills_config_path(),
            scope="user",
        )

        merged_entries = self._merge_skill_entries(
            global_entries, user_entries)

        resolved: list[dict[str, str]] = []
        seen_dirs: set[str] = set()
        for entry in merged_entries:
            enabled = entry.get("enabled", True)
            if not bool(enabled):
                continue

            skill_dir = self._resolve_skill_dir(entry)
            if skill_dir is None:
                continue

            if not (skill_dir / "SKILL.md").exists():
                continue

            dir_text = str(skill_dir)
            if dir_text in seen_dirs:
                continue
            seen_dirs.add(dir_text)

            skill_name = str(entry.get("name", "")).strip() or skill_dir.name
            source = str(entry.get("__scope", "global")
                         ).strip().lower() or "global"
            resolved.append(
                {
                    "name": skill_name,
                    "source": source,
                    "dir": dir_text,
                }
            )

        return resolved

    def _load_skill_entries(self, config_path: Path, scope: str) -> list[dict[str, Any]]:
        if not config_path.exists():
            return []
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        if not isinstance(loaded, dict):
            return []

        raw_skills = loaded.get("skills")
        entries: list[dict[str, Any]] = []

        if isinstance(raw_skills, list):
            for item in raw_skills:
                if isinstance(item, dict):
                    entry = dict(item)
                    entry["__scope"] = scope
                    entries.append(entry)
        elif isinstance(raw_skills, dict):
            for name, setting in raw_skills.items():
                if isinstance(setting, bool):
                    entries.append(
                        {"name": str(name), "enabled": setting, "__scope": scope})
                elif isinstance(setting, dict):
                    entry = dict(setting)
                    entry.setdefault("name", str(name))
                    entry["__scope"] = scope
                    entries.append(entry)

        return entries

    def _merge_skill_entries(
        self,
        global_entries: list[dict[str, Any]],
        user_entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        index_map: dict[str, int] = {}

        for entry in global_entries:
            key = self._skill_entry_key(entry)
            if key in index_map:
                continue
            index_map[key] = len(merged)
            merged.append(entry)

        for entry in user_entries:
            key = self._skill_entry_key(entry)
            if key in index_map:
                merged[index_map[key]] = entry
                continue
            index_map[key] = len(merged)
            merged.append(entry)

        return merged

    def _skill_entry_key(self, entry: dict[str, Any]) -> str:
        raw_dir = str(entry.get("dir", "")).strip()
        if raw_dir:
            return f"dir:{raw_dir}"
        name = str(entry.get("name", "")).strip()
        return f"name:{name}"

    def _resolve_skill_dir(self, entry: dict[str, Any]) -> Path | None:
        raw_dir = str(entry.get("dir", "")).strip()
        if raw_dir:
            candidate = Path(raw_dir)
            if not candidate.is_absolute():
                candidate = self._skills_root_dir() / candidate
            return candidate

        skill_name = str(entry.get("name", "")).strip()
        if not skill_name:
            return None

        scope = str(entry.get("__scope", "global")).strip().lower()
        if scope == "user":
            user_candidate = self._skills_root_dir() / self.user_id / skill_name
            if (user_candidate / "SKILL.md").exists():
                return user_candidate

            global_candidate = self._skills_root_dir() / "global" / skill_name
            if (global_candidate / "SKILL.md").exists():
                return global_candidate

            return user_candidate
        return self._skills_root_dir() / "global" / skill_name

    def _user_skills_config_path(self) -> Path:
        return self._skills_root_dir() / self.user_id / "skills.yaml"

    def _global_skills_config_path(self) -> Path:
        return self._skills_root_dir() / "global" / "skills.yaml"

    def _skills_root_dir(self) -> Path:
        return self._wozclaw_dir() / "skills"

    def _project_root_dir(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _wozclaw_dir(self, as_bash: bool = False) -> Path | str:
        configured = self._configured_wozclaw_dir()
        wozclaw_path: Path
        if configured is not None:
            wozclaw_path = configured
        else:
            wozclaw_path = self._project_root_dir() / ".wozclaw"

        if as_bash:
            return self._to_bash_path(wozclaw_path)
        return wozclaw_path

    def _to_bash_path(self, path: Path) -> str:
        resolved = path.resolve()
        posix_text = resolved.as_posix()

        drive = resolved.drive
        if not drive:
            return posix_text

        drive_letter = drive.rstrip(":").lower()
        suffix = posix_text[2:].lstrip("/")
        if suffix:
            return f"/{drive_letter}/{suffix}"
        return f"/{drive_letter}"

    def _record_tool_trace(self, name: str, input_value: Any, output_value: Any) -> None:
        item = {
            "name": str(name),
            "input": self._stringify_tool_value(input_value),
            "output": self._stringify_tool_value(output_value),
        }
        self._active_tool_traces.append(item)
        self._active_activity_traces.append(
            {
                "type": "tool",
                "name": item["name"],
                "source": "",
                "dir": "",
                "input": item["input"],
                "output": item["output"],
            }
        )
        try:
            self.memory_store.append_session_memory_tool_trace(
                self.user_id,
                self.session_id,
                item["name"],
                item["input"],
                item["output"],
            )
        except Exception:
            # Keep tracing non-blocking for agent runtime.
            print(f"Failed to record tool trace for {item['name']}")
            pass
        self._refresh_prompt_with_latest_session_memory()

    def _to_tool_response(self, text: str) -> ToolResponse:
        return ToolResponse(content=[{"type": "text", "text": text}])

    def _msg_to_text(self, msg: Any) -> str:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)
        text_chunks: list[str] = []
        thinking_chunks: list[str] = []
        for block in content:
            if isinstance(block, dict):
                block_type = str(block.get("type", "")).strip().lower()
                if block_type == "text":
                    text_value = block.get("text", "")
                    if text_value:
                        text_chunks.append(str(text_value))
                    continue
                if block_type in {"thinking", "reasoning"}:
                    thinking_value = block.get("thinking") or block.get(
                        "text") or block.get("content", "")
                    if thinking_value:
                        thinking_chunks.append(str(thinking_value))
                    continue
                content_value = block.get("content", "")
                if content_value:
                    text_chunks.append(str(content_value))
                continue

            block_type = str(getattr(block, "type", "")).strip().lower()
            if block_type in {"thinking", "reasoning"}:
                thinking_value = getattr(block, "thinking", "") or getattr(
                    block, "text", "") or getattr(block, "content", "")
                if thinking_value:
                    thinking_chunks.append(str(thinking_value))
                continue

            text_attr = getattr(block, "text", "")
            if text_attr:
                text_chunks.append(str(text_attr))
                continue

            content_attr = getattr(block, "content", "")
            if content_attr:
                text_chunks.append(str(content_attr))

        if text_chunks:
            return "\n".join(text_chunks).strip()
        if thinking_chunks:
            return "\n".join(thinking_chunks).strip()
        return ""

    def _run_async(self, coro: Any) -> Any:
        try:
            return asyncio.run(coro)
        except RuntimeError:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: asyncio.run(coro))
                return future.result()

    async def _await_with_interrupt(self, coro: Any) -> Any:
        task = asyncio.create_task(coro)
        while not task.done():
            if self._approval_interrupt_requested:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                raise ApprovalInterrupt(self._approval_interrupt_message)
            if self._choice_interrupt_requested:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                raise ChoiceInterrupt(self._choice_interrupt_message)
            await asyncio.sleep(0.01)
        return await task

    def _run_async_with_interrupt(self, coro: Any) -> Any:
        return self._run_async(self._await_with_interrupt(coro))

    def _refresh_prompt_with_latest_session_memory(self) -> None:
        """Refresh sys_prompt with latest session_memory after each tool node."""
        try:
            context = self.memory_store.load_context(
                self.user_id,
                self.session_id,
                query="",
                session_limit=0,
                session_token_budget=2800,
                daily_limit=0,
            )
            latest_session_memory = context.session_memory
            if latest_session_memory != self._active_session_memory:
                self._active_session_memory = latest_session_memory
                prompt = self.build_system_prompt(self._active_memory_context)
                self._set_runtime_prompt(prompt)
        except Exception:
            pass

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

    def _render_recent_session_messages(self, rounds: int = 3) -> str:
        try:
            rows = self.memory_store.get_recent_session_messages(
                self.user_id,
                self.session_id,
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

    def _compact_session_memory_now(self, force: bool = False) -> bool:
        if self._compact_model is None:
            return False

        state = self.memory_store.get_session_state(
            self.user_id, self.session_id)
        raw_memory = str(state.get("session_memory", "")).strip()
        if not raw_memory:
            return False

        token_count = self._estimate_text_tokens(raw_memory)
        if not force and token_count <= SESSION_MEMORY_COMPACT_THRESHOLD_TOKENS:
            return False

        prompt = self._build_compact_session_memory_prompt(raw_memory)
        response = self._run_async(
            self._compact_model(
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
        )
        compacted = self._msg_to_text(response).strip()
        if not compacted:
            return False

        self.memory_store.update_session_state(
            self.user_id,
            self.session_id,
            {"session_memory": compacted},
        )
        self._refresh_prompt_with_latest_session_memory()
        return True

    def _estimate_text_tokens(self, text: str) -> int:
        content = str(text or "")
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", content))
        other_chars = max(0, len(content) - chinese_chars)
        return max(1, chinese_chars + ((other_chars + 3) // 4))

    def _set_runtime_prompt(self, prompt: str) -> None:
        # AgentScope exposes sys_prompt as read-only; update its backing field when available.
        if hasattr(self._agent, "_sys_prompt"):
            setattr(self._agent, "_sys_prompt", prompt)
            return
        try:
            setattr(self._agent, "sys_prompt", prompt)
        except Exception:
            return

    def _stringify_tool_value(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    def build_system_prompt(self, memory_context: str) -> str:
        return (
            "<ROLE>你是一个带记忆的全能助手，会优先遵守长期记忆中的稳定偏好。</ROLE>\n"
            "<MISSION>在最少猜测下给出可验证、可追溯的回答与帮助。</MISSION>\n\n"
            "<MEMORY_RULES>\n"
            "1) 用户信息新增/变化时，可调用 remember_note。\n"
            "2) remember_note.note 必须是 memory.md 的完整新版本（非增量）。\n"
            "3) 保留仍有效旧信息并融合新信息；若冲突，以最新明确信息覆盖旧信息。\n"
            "</MEMORY_RULES>\n\n"
            "<TOOL_POLICY>\n"
            "总原则：能用工具就先用工具；可验证问题禁止猜测。\n"
            "文件相关优先使用文件工具：list_dir、file_exists、read_file、search_file、write_file、append_file、replace_file_lines、patch_file、copy_file、move_file、delete_file。\n"
            "仅当工具无法解决时，才可直接回答并说明原因。\n"
            "不确定用户偏好或方案时，调用 ask_human_choice；options 提供 2-5 个简短选项并允许自定义。\n"
            "会话变长或冗余时，主动调用 compact_context 压缩上下文。\n"
            "</TOOL_POLICY>\n\n"
            "<CONTEXT_RETRIEVAL>\n"
            "默认仅注入长期记忆；短期会话记忆通过单独用户消息提供。\n"
            "需要 session/daily 历史时先 search_session/search_daily，再用 message_id 调用 get_session_window/get_daily_window 扩展上下文。\n"
            "</CONTEXT_RETRIEVAL>\n\n"
            "<BASH_POLICY>\n"
            "需要执行命令时使用 bash_command。\n"
            "文件浏览/读写优先文件工具，不要先用 bash_command。\n"
            "长任务或仅需启动任务时使用 bash_command(background=true)。\n"
            "编辑流程：先搜索定位，再最小读取，最后最小修改；非必要不整文件替换。\n"
            "bash_command 在当前 Bash 工作目录执行，并在命令后保持目录状态；输出最多保留100000字符。\n"
            "</BASH_POLICY>\n\n"
            "<PATHS>\n"
            f"工作目录: {self._root_work_dir(as_bash=False)}\n"
            f"配置目录: {self._wozclaw_dir(as_bash=False)}\n"
            "skills 目录位于配置目录下。\n"
            "</PATHS>\n\n"
            f"<USER>当前用户ID: {self.user_id}</USER>\n\n"
            f"<MEMORY>\n{memory_context}\n</MEMORY>"
        )


class ConversationTitleGenerator:
    def __init__(self) -> None:
        self._config = load_llm_config()

    def generate_title(self, user_message: str, assistant_reply: str) -> str:
        _ = assistant_reply
        if not self._config.api_key:
            return self._fallback(user_message)

        try:
            model = self._build_chat_model()
            formatter = OpenAIChatFormatter()
            prompt = (
                "请仅根据用户第一条消息生成一个简短中文会话标题（不超过12字，不要标点和引号）。\n"
                f"用户第一条消息: {user_message}\n"
                "只输出标题本身。"
            )
            formatted_messages = self._run_async(
                formatter.format(
                    [
                        Msg(name="system", role="system", content="你是会话标题生成助手。"),
                        Msg(name="user", role="user", content=prompt),
                    ]
                )
            )
            response = self._run_async(model(formatted_messages))
            title = self._response_to_text(response).strip().replace("\n", "")
            return title[:12] if title else self._fallback(user_message)
        except Exception:
            return self._fallback(user_message)

    def _build_chat_model(self) -> OpenAIChatModel:
        client_kwargs: dict[str, Any] = {}
        if self._config.base_url.strip():
            client_kwargs["base_url"] = self._config.base_url.strip()
        return OpenAIChatModel(
            model_name=self._config.model,
            api_key=self._config.api_key,
            stream=False,
            client_kwargs=client_kwargs or None,
            generate_kwargs={"temperature": 0},
        )

    def _response_to_text(self, response: Any) -> str:
        content = getattr(response, "content", "")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)

        text_chunks: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text_value = block.get("text") if block.get(
                    "type") == "text" else block.get("content", "")
                if text_value:
                    text_chunks.append(str(text_value))
                continue
            text_attr = getattr(block, "text", "")
            if text_attr:
                text_chunks.append(str(text_attr))
        return "\n".join(text_chunks)

    def _run_async(self, coro: Any) -> Any:
        try:
            return asyncio.run(coro)
        except RuntimeError:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: asyncio.run(coro))
                return future.result()

    def _fallback(self, user_message: str) -> str:
        base = user_message.strip() or "新对话"
        return base[:12]
