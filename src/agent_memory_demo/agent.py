from __future__ import annotations

import asyncio
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import ToolResponse, Toolkit
import yaml

from agent_memory_demo.config import load_llm_config
from agent_memory_demo.memory_store import MemoryStore


@dataclass
class AgentResponse:
    text: str
    tool_calls: list[dict[str, str]] = field(default_factory=list)
    loaded_skills: list[dict[str, str]] = field(default_factory=list)
    activity_traces: list[dict[str, str]] = field(default_factory=list)


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
            self._agent = None
            self._fallback = FallbackAgent()
            self._loaded_skills: list[dict[str, str]] = []
            return
        self._active_tool_traces: list[dict[str, str]] = []
        self._active_activity_traces: list[dict[str, str]] = []
        self._loaded_skills: list[dict[str, str]] = []
        model = self._build_chat_model(
            api_key=llm_config.api_key,
            model_name=llm_config.model,
            base_url=llm_config.base_url,
            temperature=0.2,
        )
        formatter = OpenAIChatFormatter()
        toolkit = self._build_toolkit()
        self._agent = ReActAgent(
            name="memory-assistant",
            sys_prompt=self.build_system_prompt(""),
            model=model,
            formatter=formatter,
            toolkit=toolkit,
            enable_rewrite_query=False,
            print_hint_msg=False,
            max_iters=20,
        )
        self._fallback = FallbackAgent()

    def respond(self, user_message: str, memory_context: str) -> AgentResponse:
        if self._agent is None:
            return AgentResponse(
                text=self._fallback.respond(user_message, memory_context),
                tool_calls=[],
                loaded_skills=[],
                activity_traces=[],
            )

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

        try:
            reply_msg = self._run_async(
                self._agent.reply(
                    Msg(name="user", content=user_message, role="user"))
            )
        except Exception:
            return AgentResponse(text=self._fallback.respond(user_message, memory_context), tool_calls=[])

        reply_text = self._msg_to_text(reply_msg)
        if not reply_text:
            reply_text = self._fallback.respond(user_message, memory_context)

        return AgentResponse(
            text=reply_text,
            tool_calls=list(self._active_tool_traces),
            loaded_skills=list(self._loaded_skills),
            activity_traces=list(self._active_activity_traces),
        )

    def _build_chat_model(self, api_key: str, model_name: str, base_url: str, temperature: float) -> OpenAIChatModel:
        client_kwargs: dict[str, Any] = {}
        if base_url.strip():
            client_kwargs["base_url"] = base_url.strip()
        return OpenAIChatModel(
            model_name=model_name,
            api_key=api_key,
            stream=False,
            client_kwargs=client_kwargs or None,
            generate_kwargs={"temperature": temperature},
        )

    def _build_toolkit(self) -> Toolkit:
        toolkit = Toolkit()

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

        def skill_shell_command(command: str) -> ToolResponse:
            """Run a restricted shell command inside the skills root only.

            Boundaries:
            - Working directory is fixed to the project `skills/` root.
            - Use relative paths from `skills/` root (e.g. `demo-user/demo-user-style/SKILL.md`).
            - Do NOT use absolute paths, `..`, command chaining, pipes, redirection, or environment access.
            - For `Get-Content` / `Set-Content` / `Add-Content`, you must specify `-Encoding UTF8`.

            Preferred commands on Windows:
            - Read: `Get-Content demo-user/demo-user-style/SKILL.md -Encoding UTF8`
            - List: `Get-ChildItem demo-user`
            - Write: `Set-Content demo-user/demo-user-style/SKILL.md -Value "..." -Encoding UTF8`
            """
            input_payload = {"command": command}
            try:
                output = self._run_skill_shell_command(command)
            except Exception as exc:  # pragma: no cover
                output = f"error: {exc}"
            self._record_tool_trace(
                "skill_shell_command", input_payload, output)
            return self._to_tool_response(output)

        toolkit.register_tool_function(remember_note)
        toolkit.register_tool_function(get_recent_session)
        toolkit.register_tool_function(search_session)
        toolkit.register_tool_function(get_session_window)
        toolkit.register_tool_function(search_daily)
        toolkit.register_tool_function(get_daily_window)
        toolkit.register_tool_function(skill_shell_command)
        self._register_user_skills(toolkit)
        return toolkit

    def _validate_skill_shell_command(self, command: str) -> tuple[bool, str]:
        text = command.strip()
        if not text:
            return False, "empty command"

        # Disallow multi-command chaining, redirection, env access, and path traversal.
        forbidden_tokens = ["&&", "||", ";", "|", ">", "<", "`", "$env:", ".."]
        lowered = text.lower()
        for token in forbidden_tokens:
            if token in lowered:
                return False, f"unsafe token: {token}"

        # Disallow absolute paths on Windows and POSIX style root paths.
        if ":\\" in text or text.startswith("/") or " /" in text:
            return False, "unsafe absolute path"

        first = text.split(maxsplit=1)[0].strip().lower()
        allowed = {
            "get-childitem",
            "get-content",
            "set-content",
            "add-content",
            "new-item",
            "remove-item",
            "copy-item",
            "move-item",
            "test-path",
            "select-string",
            "ls",
            "cat",
            "echo",
            "mkdir",
        }
        if first not in allowed:
            return False, f"unsafe command: {first}"

        if first in {"get-content", "set-content", "add-content", "cat"}:
            if re.search(r"-encoding\s+utf-?8\b", lowered) is None:
                return False, "encoding must be UTF8"

        return True, ""

    def _run_skill_shell_command(self, command: str) -> str:
        ok, reason = self._validate_skill_shell_command(command)
        if not ok:
            return (
                "error: unsafe skill shell command "
                f"({reason}). Use relative paths from skills root, such as "
                "`Get-Content demo-user/demo-user-style/SKILL.md -Encoding UTF8`."
            )

        skills_root = self._skills_root_dir()
        skills_root.mkdir(parents=True, exist_ok=True)

        if os.name == "nt":
            powershell_prefix = (
                "$OutputEncoding = [System.Text.Encoding]::UTF8; "
                "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            )
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                f"{powershell_prefix}& {{ {command} }}",
            ]
        else:
            cmd = ["sh", "-lc", command]

        result = subprocess.run(
            cmd,
            cwd=skills_root,
            capture_output=True,
            text=True,
            timeout=15,
            shell=False,
            encoding="utf-8",
            errors="replace",
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if result.returncode != 0:
            error_text = stderr or stdout or f"command failed with exit code {result.returncode}"
            return f"error: {error_text}"

        return stdout[:8000] if stdout else "ok"

    def _register_user_skills(self, toolkit: Toolkit) -> None:
        self._loaded_skills = []
        for item in self._resolve_enabled_skills():
            skill_dir = Path(item.get("dir", ""))
            if not str(skill_dir):
                continue
            try:
                toolkit.register_agent_skill(str(skill_dir))
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
        project_root = Path(__file__).resolve().parents[2]
        return project_root / "skills"

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

    def _to_tool_response(self, text: str) -> ToolResponse:
        return ToolResponse(content=[{"type": "text", "text": text}])

    def _msg_to_text(self, msg: Any) -> str:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)
        chunks: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text_value = block.get("text") if block.get(
                    "type") == "text" else block.get("content", "")
                if text_value:
                    chunks.append(str(text_value))
                continue
            text_attr = getattr(block, "text", "")
            if text_attr:
                chunks.append(str(text_attr))
        return "\n".join(chunks).strip()

    def _run_async(self, coro: Any) -> Any:
        try:
            return asyncio.run(coro)
        except RuntimeError:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: asyncio.run(coro))
                return future.result()

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
            "当需要读取或修改技能文件时，必须使用 skill_shell_command。\n"
            "skill_shell_command 的边界：只能在 skills 根目录下执行；路径必须是相对路径；禁止绝对路径、..、管道、重定向和多命令拼接。\n"
            "读取 demo-user-style 的正确示例：Get-Content demo-user/demo-user-style/SKILL.md -Encoding UTF8\n"
            "写入技能文件也必须显式带 -Encoding UTF8，例如 Set-Content ... -Encoding UTF8。\n"
            "不要把路径写成 skills/demo-user/...，因为工具的当前目录已经是 skills/。\n\n"
            f"当前用户ID: {self.user_id}\n\n"
            f"可用记忆上下文:\n{memory_context}"
        )


class ConversationTitleGenerator:
    def __init__(self) -> None:
        self._config = load_llm_config()

    def generate_title(self, user_message: str, assistant_reply: str) -> str:
        if not self._config.api_key:
            return self._fallback(user_message)

        try:
            model = self._build_chat_model()
            formatter = OpenAIChatFormatter()
            prompt = (
                "请根据以下对话生成一个简短中文会话标题（不超过12字，不要标点和引号）。\n"
                f"用户: {user_message}\n"
                f"助手: {assistant_reply}\n"
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
