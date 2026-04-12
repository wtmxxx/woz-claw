from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


DEFAULT_COMMAND_POLICY: dict[str, Any] = {
    "enabled": True,
    "default_action": "ask_human",
    "operations": {
        "read": "allow",
        "write": "ask_human",
        "delete": "ask_human",
        "exec": "ask_human",
    },
    "allowed_paths": [],
    "command_allowlist": [],
    "command_blocklist": [],
}


@dataclass
class MemoryContext:
    session_messages: list[dict[str, Any]]
    daily_messages: list[dict[str, Any]]
    long_term_hits: list[dict[str, Any]]
    session_memory: str = ""


class MemoryStore:
    def __init__(
        self,
        root_dir: Path | str = "memory",
        command_policy_root_dir: Path | str | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if command_policy_root_dir is None:
            if self.root_dir.name.lower() == "memory":
                self.command_policy_root_dir = self.root_dir.parent / "config"
            else:
                self.command_policy_root_dir = self.root_dir / "config"
        else:
            self.command_policy_root_dir = Path(command_policy_root_dir)

    def append_session_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
        record_session_memory: bool = True,
    ) -> None:
        file_path = self._session_file(user_id, session_id)
        next_message_id = self._next_message_id(self._read_jsonl(file_path))
        payload = self._message_payload(
            role=role, content=content, tags=["session"], meta=meta)
        payload["message_id"] = next_message_id
        self._append_jsonl(file_path, payload)
        if record_session_memory:
            self._append_session_memory_entry(user_id, session_id, payload)

    def append_session_memory_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        payload = {
            "role": role,
            "content": content,
        }
        self._append_session_memory_entry(user_id, session_id, payload)

    def append_daily_message(
        self,
        user_id: str,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        file_path = self._daily_file(user_id)
        next_message_id = self._next_message_id(self._read_jsonl(file_path))
        payload = self._message_payload(
            role=role, content=content, tags=["daily"], meta=meta)
        payload["message_id"] = next_message_id
        self._append_jsonl(file_path, payload)

    def remember_long_term(self, user_id: str, note: str, tags: list[str] | None = None) -> None:
        _ = tags
        normalized_note = note.strip()
        if not normalized_note:
            return
        # The note is treated as the latest full memory summary from the LLM.
        self.set_long_term_memory(user_id, normalized_note)

    def append_llm_dialogue_log(
        self,
        user_id: str,
        session_id: str,
        input_value: Any,
        output_value: Any,
    ) -> None:
        payload = {
            "input": input_value,
            "output": output_value,
        }
        self._append_jsonl(self._llm_log_file(user_id), payload)

    def get_session_messages(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        return self._normalize_message_rows(self._read_jsonl(self._session_file(user_id, session_id)))

    def get_recent_session_messages(self, user_id: str, session_id: str, rounds: int = 3) -> list[dict[str, Any]]:
        rows = self.get_session_messages(user_id, session_id)
        limit = max(1, rounds) * 2
        return rows[-limit:]

    def search_session_messages(self, user_id: str, session_id: str, keyword: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.get_session_messages(user_id, session_id)
        return self._search_rows(rows, keyword=keyword, limit=limit)

    def get_session_messages_window(
        self,
        user_id: str,
        session_id: str,
        message_id: int,
        before: int = 0,
        after: int = 0,
    ) -> list[dict[str, Any]]:
        rows = self.get_session_messages(user_id, session_id)
        anchor_index = self._find_message_index(rows, message_id)
        if anchor_index is None:
            return []

        safe_before = max(0, before)
        safe_after = max(0, after)
        start = max(0, anchor_index - safe_before)
        end = min(len(rows), anchor_index + safe_after + 1)
        return rows[start:end]

    def get_daily_messages_by_date(self, user_id: str, day: str) -> list[dict[str, Any]]:
        date_text = day.strip()
        if not date_text:
            date_text = datetime.now().date().isoformat()
        path = self._user_root(user_id) / "daily" / f"{date_text}.jsonl"
        return self._normalize_message_rows(self._read_jsonl(path))

    def search_daily_messages(self, user_id: str, keyword: str, limit: int = 20, day: str = "") -> list[dict[str, Any]]:
        daily_dir = self._user_root(user_id) / "daily"
        if not daily_dir.exists():
            return []

        rows: list[dict[str, Any]] = []
        day_text = day.strip()
        if day_text:
            try:
                datetime.fromisoformat(day_text)
                file = daily_dir / f"{day_text}.jsonl"
                rows.extend(self._normalize_message_rows(
                    self._read_jsonl(file)))
            except ValueError:
                # Invalid day format falls back to full daily search.
                for file in sorted(daily_dir.glob("*.jsonl")):
                    rows.extend(self._normalize_message_rows(
                        self._read_jsonl(file)))
        else:
            for file in sorted(daily_dir.glob("*.jsonl")):
                rows.extend(self._normalize_message_rows(
                    self._read_jsonl(file)))
        return self._search_rows(rows, keyword=keyword, limit=limit)

    def get_daily_messages_window(
        self,
        user_id: str,
        message_id: int,
        before: int = 0,
        after: int = 0,
        day: str = "",
    ) -> list[dict[str, Any]]:
        rows = self.get_daily_messages_by_date(user_id, day)
        anchor_index = self._find_message_index(rows, message_id)
        if anchor_index is None:
            return []

        safe_before = max(0, before)
        safe_after = max(0, after)
        start = max(0, anchor_index - safe_before)
        end = min(len(rows), anchor_index + safe_after + 1)
        return rows[start:end]

    def get_conversation_title(self, user_id: str, session_id: str) -> str | None:
        rows = self._read_json(self._conversation_file(user_id))
        value = rows.get(session_id)
        if not isinstance(value, dict):
            return None
        title = value.get("title")
        return str(title) if title else None

    def set_conversation_title(self, user_id: str, session_id: str, title: str) -> None:
        rows = self._read_json(self._conversation_file(user_id))
        rows[session_id] = {
            "title": title.strip() or "新对话",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._write_json(self._conversation_file(user_id), rows)

    def list_conversations(self, user_id: str) -> list[dict[str, str]]:
        rows = self._read_json(self._conversation_file(user_id))
        result: list[dict[str, str]] = []
        for session_id, value in rows.items():
            if not isinstance(value, dict):
                continue
            title = str(value.get("title") or "新对话")
            updated_at = str(value.get("updated_at") or "")
            if not updated_at:
                updated_at = datetime.now().isoformat(timespec="seconds")
            result.append(
                {
                    "session_id": str(session_id),
                    "title": title,
                    "updated_at": updated_at,
                }
            )

        result.sort(key=lambda item: item["updated_at"], reverse=True)
        return result

    def delete_conversation(self, user_id: str, session_id: str) -> bool:
        user_text = user_id.strip()
        session_text = session_id.strip()
        if not user_text or not session_text:
            return False

        removed_any = False

        conv_path = self._conversation_file(user_text)
        rows = self._read_json(conv_path)
        if session_text in rows:
            rows.pop(session_text, None)
            self._write_json(conv_path, rows)
            removed_any = True

        for path in [
            self._session_file(user_text, session_text),
            self._session_state_file(user_text, session_text),
            self._approvals_file(user_text, session_text),
            self._choices_file(user_text, session_text),
            self._react_state_file(user_text, session_text),
        ]:
            if path.exists():
                path.unlink()
                removed_any = True

        if self._delete_daily_messages_by_session(user_text, session_text):
            removed_any = True

        return removed_any

    def get_long_term_memory(self, user_id: str) -> str:
        path = self._long_term_file(user_id)
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def set_long_term_memory(self, user_id: str, content: str) -> None:
        path = self._long_term_file(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized = content.strip()
        if normalized:
            path.write_text(normalized + "\n", encoding="utf-8")
        else:
            path.write_text("", encoding="utf-8")

    def remove_long_term_memory(self, user_id: str, snippet: str) -> bool:
        target = snippet.strip()
        if not target:
            return False

        content = self.get_long_term_memory(user_id)
        if not content:
            return False

        lines = content.splitlines()
        remained = [line for line in lines if target not in line]
        if len(remained) == len(lines):
            return False
        self.set_long_term_memory(user_id, "\n".join(remained))
        return True

    def get_user_settings(self, user_id: str) -> dict[str, Any]:
        return self._read_json(self._settings_file(user_id))

    def set_user_settings(self, user_id: str, settings: dict[str, Any]) -> None:
        normalized = settings if isinstance(settings, dict) else {}
        self._write_json(self._settings_file(user_id), normalized)

    def get_command_policy(self, user_id: str) -> dict[str, Any]:
        stored_policy = self._read_json(self._command_policy_file(user_id))
        if not stored_policy:
            stored_policy = self._read_json(
                self._legacy_command_policy_file(user_id))
        if isinstance(stored_policy, dict) and stored_policy:
            return self._merge_command_policy(stored_policy)

        settings = self.get_user_settings(user_id)
        raw_policy = settings.get("command_policy") if isinstance(
            settings, dict) else {}
        policy = self._merge_command_policy(
            raw_policy if isinstance(raw_policy, dict) else {})
        return policy

    def set_command_policy(self, user_id: str, policy: dict[str, Any]) -> None:
        self._write_json(
            self._command_policy_file(user_id),
            self._merge_command_policy(policy),
        )

    def create_pending_approval(
        self,
        user_id: str,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        path = self._approvals_file(user_id, session_id)
        rows = self._read_json(path)
        request_id = uuid4().hex[:12]
        item = {
            "request_id": request_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            **(payload if isinstance(payload, dict) else {}),
        }
        rows[request_id] = item
        self._write_json(path, rows)
        return item

    def pop_pending_approval(self, user_id: str, session_id: str, request_id: str) -> dict[str, Any] | None:
        req = request_id.strip()
        if not req:
            return None
        path = self._approvals_file(user_id, session_id)
        rows = self._read_json(path)
        value = rows.pop(req, None)
        self._write_json(path, rows)
        return value if isinstance(value, dict) else None

    def replace_approval_placeholder_output(
        self,
        user_id: str,
        session_id: str,
        request_id: str,
        output: str,
    ) -> bool:
        req = request_id.strip()
        if not req:
            return False

        replaced = False
        session_path = self._session_file(user_id, session_id)
        if self._replace_placeholder_in_jsonl(
            session_path,
            req,
            output,
            marker="__APPROVAL_REQUIRED__",
            expected_tool_name="bash_command",
        ):
            replaced = True

        daily_path = self._daily_file(user_id)
        if self._replace_placeholder_in_jsonl(
            daily_path,
            req,
            output,
            marker="__APPROVAL_REQUIRED__",
            expected_tool_name="bash_command",
        ):
            replaced = True

        return replaced

    def replace_choice_placeholder_output(
        self,
        user_id: str,
        session_id: str,
        request_id: str,
        output: str,
    ) -> bool:
        req = request_id.strip()
        if not req:
            return False

        replaced = False
        session_path = self._session_file(user_id, session_id)
        if self._replace_placeholder_in_jsonl(
            session_path,
            req,
            output,
            marker="__CHOICE_REQUIRED__",
            expected_tool_name="ask_human_choice",
        ):
            replaced = True

        daily_path = self._daily_file(user_id)
        if self._replace_placeholder_in_jsonl(
            daily_path,
            req,
            output,
            marker="__CHOICE_REQUIRED__",
            expected_tool_name="ask_human_choice",
        ):
            replaced = True

        return replaced

    def create_pending_choice(
        self,
        user_id: str,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        path = self._choices_file(user_id, session_id)
        rows = self._read_json(path)
        raw_payload = payload if isinstance(payload, dict) else {}
        request_id = str(raw_payload.get("request_id", "")
                         ).strip() or uuid4().hex[:12]
        item = {
            "request_id": request_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            **raw_payload,
        }
        rows[request_id] = item
        self._write_json(path, rows)
        return item

    def pop_pending_choice(self, user_id: str, session_id: str, request_id: str) -> dict[str, Any] | None:
        req = request_id.strip()
        if not req:
            return None
        path = self._choices_file(user_id, session_id)
        rows = self._read_json(path)
        value = rows.pop(req, None)
        self._write_json(path, rows)
        return value if isinstance(value, dict) else None

    def get_pending_approvals(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        """Get all pending approvals for a session without removing them."""
        path = self._approvals_file(user_id, session_id)
        rows = self._read_json(path)
        return [item for item in rows.values() if isinstance(item, dict)]

    def get_pending_choices(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        """Get all pending choices for a session without removing them."""
        path = self._choices_file(user_id, session_id)
        rows = self._read_json(path)
        return [item for item in rows.values() if isinstance(item, dict)]

    def get_session_state(self, user_id: str, session_id: str) -> dict[str, Any]:
        path = self._session_state_file(user_id, session_id)
        data = self._read_json(path)
        return data if isinstance(data, dict) else {}

    def update_session_state(self, user_id: str, session_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        path = self._session_state_file(user_id, session_id)
        current = self._read_json(path)
        if not isinstance(current, dict):
            current = {}
        if isinstance(patch, dict):
            current.update(patch)
        self._write_json(path, current)
        return current

    def append_session_memory_tool_trace(
        self,
        user_id: str,
        session_id: str,
        name: str,
        tool_input: str = "",
        tool_output: str = "",
    ) -> None:
        """Append a tool trace to session memory, respecting filter rules."""
        approval_marker = "__APPROVAL_REQUIRED__"
        choice_required_marker = "__CHOICE_REQUIRED__"
        approval_error_prefix = "Error: 检测到需要人工审批"
        choice_error_prefix = "Error: 检测到需要做出选择"

        tool_name = str(name).strip()
        tool_output_str = str(tool_output).strip()
        if tool_name == "bash_command" and (
            tool_output_str.startswith(approval_marker)
            or tool_output_str.startswith(approval_error_prefix)
        ):
            return
        if tool_name == "ask_human_choice" and tool_output_str.startswith(choice_error_prefix):
            return
        if tool_name == "ask_human_choice" and tool_output_str.startswith(choice_required_marker):
            tool_output_str = ""

        lines: list[str] = []
        lines.append(f"[tool] {tool_name}")
        tool_input_str = str(tool_input).strip() if tool_input else ""
        if tool_name == "compact_context":
            # compact_context summary text can be very long and noisy; persist only tool output.
            tool_input_str = ""
        if tool_input_str:
            lines.append(f"输入: {tool_input_str}")
        if tool_output_str:
            lines.append(f"输出: {tool_output_str}")

        # Nothing useful to persist after filtering.
        if not tool_input_str and not tool_output_str:
            return

        entry = "\n".join(lines)
        state = self.get_session_state(user_id, session_id)
        previous = str(state.get("session_memory", "")).strip()
        if previous.endswith(entry):
            return
        merged = f"{previous}\n{entry}".strip() if previous else entry
        self.update_session_state(
            user_id,
            session_id,
            {"session_memory": merged},
        )

    def set_pending_react_state(
        self,
        user_id: str,
        session_id: str,
        request_id: str,
        payload: dict[str, Any],
    ) -> None:
        req = request_id.strip()
        if not req:
            return
        data = payload if isinstance(payload, dict) else {}
        self.update_session_state(
            user_id,
            session_id,
            {
                "pending_approval_request_id": req,
                "approval_resume_state": data,
            },
        )

    def pop_pending_react_state(self, user_id: str, session_id: str, request_id: str) -> dict[str, Any] | None:
        req = request_id.strip()
        if not req:
            return None
        state = self.get_session_state(user_id, session_id)
        current_req = str(state.get("pending_approval_request_id", "")).strip()
        if current_req != req:
            return None

        value = state.get("approval_resume_state")
        self.update_session_state(
            user_id,
            session_id,
            {
                "pending_approval_request_id": "",
                "approval_resume_state": {},
            },
        )
        return value if isinstance(value, dict) else None

    def load_context(
        self,
        user_id: str,
        session_id: str,
        query: str,
        session_limit: int = 8,
        session_token_budget: int = 2800,
        daily_limit: int = 10,
        long_term_top_k: int = 3,
    ) -> MemoryContext:
        all_session_messages = self.get_session_messages(user_id, session_id)
        all_daily_messages = self.get_daily_messages_by_date(
            user_id, datetime.now().date().isoformat())
        # Session context now defaults to token-budget selection (<3k tokens).
        # Keep session_limit<=0 as an explicit opt-out for compatibility in tests/callers.
        if session_limit <= 0 or session_token_budget <= 0:
            session_messages = []
        else:
            session_messages = self._take_recent_messages_by_token_budget(
                all_session_messages,
                token_budget=session_token_budget,
            )
        if daily_limit <= 0:
            daily_messages = []
        else:
            daily_messages = all_daily_messages[-daily_limit:]
        _ = long_term_top_k
        long_term_text = self.get_long_term_memory(user_id)
        long_term_hits: list[dict[str, Any]] = []
        if long_term_text:
            long_term_hits = [{"content": long_term_text}]
        session_state = self.get_session_state(user_id, session_id)
        session_memory = str(session_state.get("session_memory", "")).strip()
        return MemoryContext(
            session_messages=session_messages,
            daily_messages=daily_messages,
            long_term_hits=long_term_hits,
            session_memory=session_memory,
        )

    def build_prompt_context(self, memory_context: MemoryContext) -> str:
        lines: list[str] = []
        # Prompt context now only includes long-term memory and session summary.
        if memory_context.long_term_hits:
            lines.append("[LONG_TERM]")
            for row in memory_context.long_term_hits:
                lines.append(f"{row['content']}")

        if memory_context.session_memory:
            lines.append("[SESSION_MEMORY]")
            lines.append(memory_context.session_memory)

        return "\n".join(lines)

    def build_long_term_prompt_context(self, memory_context: MemoryContext) -> str:
        lines: list[str] = []
        if memory_context.long_term_hits:
            lines.append("[LONG_TERM]")
            for row in memory_context.long_term_hits:
                lines.append(f"{row['content']}")
        return "\n".join(lines)

    def _take_recent_messages_by_token_budget(
        self,
        rows: list[dict[str, Any]],
        token_budget: int,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []

        safe_budget = max(1, token_budget)
        selected: list[dict[str, Any]] = []
        used_tokens = 0

        # Keep latest messages first under budget, then reverse back to chronological order.
        for row in reversed(rows):
            row_tokens = self._estimate_message_tokens(row)
            if selected and used_tokens + row_tokens > safe_budget:
                break
            selected.append(row)
            used_tokens += row_tokens

        selected.reverse()
        return selected

    def _estimate_message_tokens(self, row: dict[str, Any]) -> int:
        role = str(row.get("role", ""))
        content = str(row.get("content", ""))
        text = f"{role}: {content}"

        # Rough multilingual estimate: Chinese chars are often ~1 token, other text ~4 chars/token.
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        other_chars = max(0, len(text) - chinese_chars)
        return max(1, chinese_chars + math.ceil(other_chars / 4))

    def _tail_text_by_token_budget(self, text: str, token_budget: int) -> str:
        raw = str(text or "")
        if not raw.strip():
            return ""

        safe_budget = max(1, token_budget)
        if self._estimate_text_tokens(raw) <= safe_budget:
            return raw.strip()

        lines = [line for line in raw.splitlines() if line.strip()]
        if not lines:
            return ""

        selected: list[str] = []
        used = 0
        for line in reversed(lines):
            cost = self._estimate_text_tokens(line)
            if selected and used + cost > safe_budget:
                break
            selected.append(line)
            used += cost
        selected.reverse()
        return "\n".join(selected).strip()

    def _estimate_text_tokens(self, text: str) -> int:
        content = str(text or "")
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", content))
        other_chars = max(0, len(content) - chinese_chars)
        return max(1, chinese_chars + math.ceil(other_chars / 4))

    def _message_payload(
        self,
        role: str,
        content: str,
        tags: list[str],
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "role": role,
            "content": content,
            "tags": tags,
            "meta": meta if isinstance(meta, dict) else {},
        }

    def _user_root(self, user_id: str) -> Path:
        return self.root_dir / user_id

    def _session_file(self, user_id: str, session_id: str) -> Path:
        return self._user_root(user_id) / "sessions" / f"{session_id}.jsonl"

    def _daily_file(self, user_id: str) -> Path:
        day = datetime.now().date().isoformat()
        return self._user_root(user_id) / "daily" / f"{day}.jsonl"

    def _long_term_file(self, user_id: str) -> Path:
        return self._user_root(user_id) / "memory.md"

    def _llm_log_file(self, user_id: str) -> Path:
        return self._user_root(user_id) / "logs" / "llm_dialogue.jsonl"

    def _settings_file(self, user_id: str) -> Path:
        return self._user_root(user_id) / "settings.json"

    def _approvals_file(self, user_id: str, session_id: str) -> Path:
        return self._user_root(user_id) / "approvals" / f"{session_id}.json"

    def _choices_file(self, user_id: str, session_id: str) -> Path:
        return self._user_root(user_id) / "choices" / f"{session_id}.json"

    def _session_state_file(self, user_id: str, session_id: str) -> Path:
        return self._user_root(user_id) / "session_memory" / f"{session_id}.json"

    def _react_state_file(self, user_id: str, session_id: str) -> Path:
        return self._user_root(user_id) / "react_state" / f"{session_id}.json"

    def _command_policy_file(self, user_id: str) -> Path:
        return self.command_policy_root_dir / user_id / "command_policy.json"

    def _legacy_command_policy_file(self, user_id: str) -> Path:
        return self.command_policy_root_dir / "command_policies" / f"{user_id}.json"

    def _conversation_file(self, user_id: str) -> Path:
        return self._user_root(user_id) / "conversations.json"

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _append_session_memory_entry(self, user_id: str, session_id: str, message: dict[str, Any]) -> None:
        """Append user/assistant message to session memory in chronological order."""
        if not isinstance(message, dict):
            return

        role = str(message.get("role", "")).strip() or "assistant"
        content = str(message.get("content", "")).strip()

        if not content:
            return

        entry = f"{role}: {content}"
        state = self.get_session_state(user_id, session_id)
        previous = str(state.get("session_memory", "")).strip()
        merged = f"{previous}\n{entry}".strip() if previous else entry
        self.update_session_state(
            user_id,
            session_id,
            {"session_memory": merged},
        )

    def _next_message_id(self, rows: list[dict[str, Any]]) -> int:
        next_message_id = 1
        for row in rows:
            message_id = self._parse_message_id(row.get("message_id"))
            if message_id is None:
                message_id = next_message_id
            next_message_id = max(next_message_id, message_id + 1)
        return next_message_id

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _delete_daily_messages_by_session(self, user_id: str, session_id: str) -> bool:
        daily_dir = self._user_root(user_id) / "daily"
        if not daily_dir.exists():
            return False

        removed = False
        for file in sorted(daily_dir.glob("*.jsonl")):
            rows = self._read_jsonl(file)
            if not rows:
                continue

            kept_rows: list[dict[str, Any]] = []
            changed = False
            for row in rows:
                meta = row.get("meta") if isinstance(row, dict) else None
                row_session_id = ""
                if isinstance(meta, dict):
                    row_session_id = str(meta.get("session_id", "")).strip()

                if row_session_id == session_id:
                    changed = True
                    continue
                kept_rows.append(row)

            if not changed:
                continue

            removed = True
            if kept_rows:
                self._write_jsonl(file, kept_rows)
            else:
                file.unlink()

        return removed

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                if not isinstance(row, dict):
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _replace_placeholder_in_jsonl(
        self,
        path: Path,
        request_id: str,
        output: str,
        marker: str,
        expected_tool_name: str,
    ) -> bool:
        rows = self._read_jsonl(path)
        if not rows:
            return False

        for row_index in range(len(rows) - 1, -1, -1):
            row = rows[row_index]
            if str(row.get("role", "")) != "assistant":
                continue

            meta = row.get("meta")
            if not isinstance(meta, dict):
                continue

            raw_tool_calls = meta.get("tool_calls")
            if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
                continue

            changed = False
            next_tool_calls: list[dict[str, Any]] = []
            for item in raw_tool_calls:
                if not isinstance(item, dict):
                    next_tool_calls.append(item)
                    continue

                item_tool_name = str(item.get("name", ""))
                tool_output = str(item.get("output", ""))
                if item_tool_name != expected_tool_name or not tool_output.startswith(marker):
                    next_tool_calls.append(item)
                    continue

                payload_text = tool_output[len(marker):].strip()
                try:
                    payload = json.loads(payload_text)
                except Exception:
                    next_tool_calls.append(item)
                    continue

                if str(payload.get("request_id", "")).strip() != request_id:
                    next_tool_calls.append(item)
                    continue

                updated = dict(item)
                updated["output"] = output
                next_tool_calls.append(updated)
                changed = True

            if not changed:
                continue

            next_meta = dict(meta)
            next_meta["tool_calls"] = next_tool_calls

            raw_activity_traces = meta.get("activity_traces")
            if isinstance(raw_activity_traces, list) and raw_activity_traces:
                next_activity_traces: list[dict[str, Any]] = []
                for trace in raw_activity_traces:
                    if not isinstance(trace, dict):
                        next_activity_traces.append(trace)
                        continue

                    trace_type = str(trace.get("type", ""))
                    trace_name = str(trace.get("name", ""))
                    trace_output = str(trace.get("output", ""))
                    if trace_type == "tool" and trace_name == expected_tool_name and trace_output.startswith(marker):
                        payload_text = trace_output[len(marker):].strip()
                        try:
                            payload = json.loads(payload_text)
                        except Exception:
                            next_activity_traces.append(trace)
                            continue
                        if str(payload.get("request_id", "")).strip() != request_id:
                            next_activity_traces.append(trace)
                            continue
                        updated_trace = dict(trace)
                        updated_trace["output"] = output
                        next_activity_traces.append(updated_trace)
                    else:
                        next_activity_traces.append(trace)
                next_meta["activity_traces"] = next_activity_traces

            next_row = dict(row)
            next_row["meta"] = next_meta
            rows[row_index] = next_row
            self._write_jsonl(path, rows)
            return True

        return False

    def _normalize_message_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized_rows: list[dict[str, Any]] = []
        next_message_id = 1
        for row in rows:
            item = dict(row)
            message_id = self._parse_message_id(item.get("message_id"))
            if message_id is None:
                message_id = next_message_id
            item["message_id"] = message_id
            next_message_id = max(next_message_id, message_id + 1)
            normalized_rows.append(item)
        return normalized_rows

    def _parse_message_id(self, value: Any) -> int | None:
        try:
            message_id = int(value)
        except (TypeError, ValueError):
            return None
        return message_id if message_id > 0 else None

    def _find_message_index(self, rows: list[dict[str, Any]], message_id: int) -> int | None:
        target_id = self._parse_message_id(message_id)
        if target_id is None:
            return None

        for index, row in enumerate(rows):
            row_message_id = self._parse_message_id(row.get("message_id"))
            if row_message_id == target_id:
                return index
        return None

    def _read_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return raw if isinstance(raw, dict) else {}

    def _write_json(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False,
                        indent=2), encoding="utf-8")

    def _search_rows(self, rows: list[dict[str, Any]], keyword: str, limit: int) -> list[dict[str, Any]]:
        key = keyword.strip().lower()
        safe_limit = max(1, min(limit, 200))

        if not key:
            return rows[-safe_limit:]

        matched = [row for row in rows if key in str(
            row.get("content", "")).lower()]
        return matched[-safe_limit:]

    def _keyword_retrieve(self, rows: list[dict[str, Any]], query: str, top_k: int) -> list[dict[str, Any]]:
        query_tokens = set(self._tokens(query))
        query_text = query.strip().lower()
        if not query_tokens:
            return rows[-top_k:]

        scored: list[tuple[int, dict[str, Any]]] = []
        for row in rows:
            content_text = str(row.get("content", "")).lower()
            content_tokens = set(self._tokens(content_text))
            score = len(query_tokens & content_tokens)
            if query_text and query_text in content_text:
                score += 1
            scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for score, row in scored if score > 0][:top_k]

    def _tokens(self, text: str) -> list[str]:
        return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())

    def _merge_command_policy(self, policy: dict[str, Any]) -> dict[str, Any]:
        merged = deepcopy(DEFAULT_COMMAND_POLICY)
        if not isinstance(policy, dict):
            return merged

        if "enabled" in policy:
            merged["enabled"] = bool(policy.get("enabled", True))

        if isinstance(policy.get("default_action"), str):
            merged["default_action"] = str(
                policy.get("default_action") or "ask_human")

        operations = policy.get("operations")
        if isinstance(operations, dict):
            for key in ["read", "write", "delete", "exec"]:
                if isinstance(operations.get(key), str) and operations.get(key):
                    merged["operations"][key] = str(operations[key])

        allowlist = policy.get("command_allowlist")
        if isinstance(allowlist, list):
            merged["command_allowlist"] = [
                str(item).strip() for item in allowlist if str(item).strip()]

        blocklist = policy.get("command_blocklist")
        if isinstance(blocklist, list):
            merged["command_blocklist"] = [
                str(item).strip() for item in blocklist if str(item).strip()]

        allowed_paths = policy.get("allowed_paths")
        if isinstance(allowed_paths, list):
            merged["allowed_paths"] = [
                str(item).strip() for item in allowed_paths if str(item).strip()]

        return merged
