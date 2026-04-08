from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class MemoryContext:
    session_messages: list[dict[str, Any]]
    daily_messages: list[dict[str, Any]]
    long_term_hits: list[dict[str, Any]]


class MemoryStore:
    def __init__(self, root_dir: Path | str = "memory") -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def append_session_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        file_path = self._session_file(user_id, session_id)
        next_message_id = self._next_message_id(self._read_jsonl(file_path))
        payload = self._message_payload(
            role=role, content=content, tags=["session"], meta=meta)
        payload["message_id"] = next_message_id
        self._append_jsonl(file_path, payload)

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
        user_message: str,
        assistant_reply: str,
        memory_context: str,
        tool_calls: list[dict[str, str]] | None = None,
    ) -> None:
        payload = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
            "user_message": user_message,
            "assistant_reply": assistant_reply,
            "memory_context": memory_context,
            "tool_calls": tool_calls or [],
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

    def load_context(
        self,
        user_id: str,
        session_id: str,
        query: str,
        session_limit: int = 8,
        daily_limit: int = 10,
        long_term_top_k: int = 3,
    ) -> MemoryContext:
        all_session_messages = self.get_session_messages(user_id, session_id)
        all_daily_messages = self.get_daily_messages_by_date(
            user_id, datetime.now().date().isoformat())
        if session_limit <= 0:
            session_messages = []
        else:
            session_messages = all_session_messages[-session_limit:]
        if daily_limit <= 0:
            daily_messages = []
        else:
            daily_messages = all_daily_messages[-daily_limit:]
        _ = long_term_top_k
        long_term_text = self.get_long_term_memory(user_id)
        long_term_hits: list[dict[str, Any]] = []
        if long_term_text:
            long_term_hits = [{"content": long_term_text}]
        return MemoryContext(
            session_messages=session_messages,
            daily_messages=daily_messages,
            long_term_hits=long_term_hits,
        )

    def build_prompt_context(self, memory_context: MemoryContext) -> str:
        lines: list[str] = []
        if memory_context.session_messages:
            lines.append("[SESSION]")
            for row in memory_context.session_messages:
                lines.append(f"{row['role']}: {row['content']}")

        if memory_context.daily_messages:
            lines.append("[DAILY]")
            for row in memory_context.daily_messages:
                lines.append(f"{row['role']}: {row['content']}")

        if memory_context.long_term_hits:
            lines.append("[LONG_TERM]")
            for row in memory_context.long_term_hits:
                lines.append(f"{row['content']}")

        return "\n".join(lines)

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

    def _conversation_file(self, user_id: str) -> Path:
        return self._user_root(user_id) / "conversations.json"

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

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
