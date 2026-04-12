from io import BytesIO
from fastapi.testclient import TestClient
from types import SimpleNamespace
from pathlib import Path
from zipfile import ZipFile

import yaml

from wozclaw import app as app_module
from wozclaw.memory_store import MemoryStore
from wozclaw.service import ChatService


def test_get_conversation_messages_accepts_numeric_message_id(monkeypatch) -> None:
    client = TestClient(app_module.app)

    def fake_get_session_messages(user_id: str, session_id: str):
        _ = user_id
        _ = session_id
        return [
            {
                "ts": "2026-04-08T10:00:00",
                "role": "user",
                "content": "hello",
                "message_id": 1,
            }
        ]

    monkeypatch.setattr(
        app_module.chat_service,
        "get_session_messages",
        fake_get_session_messages,
    )

    response = client.get("/api/conversations/s1/messages",
                          params={"user_id": "u1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"][0]["message_id"] == 1


def test_chat_api_returns_tool_calls(monkeypatch) -> None:
    client = TestClient(app_module.app)

    def fake_chat(user_id: str, session_id: str, message: str, llm_user_message: str | None = None):
        _ = user_id
        _ = session_id
        _ = message
        _ = llm_user_message
        return SimpleNamespace(
            reply="ok",
            memory_hits=1,
            title="标题",
            tool_calls=[{"name": "search_session",
                         "input": "秘密", "output": "#3 ..."}],
            loaded_skills=[
                {"name": "memory-tools", "source": "global", "dir": "skills/global/memory-tools"}],
            activity_traces=[
                {"type": "skill", "name": "memory-tools", "source": "global",
                    "dir": "skills/global/memory-tools", "input": "", "output": "loaded"},
                {"type": "tool", "name": "search_session", "source": "",
                    "dir": "", "input": "秘密", "output": "#3 ..."},
            ],
            approval_request=None,
            choice_request=None,
        )

    monkeypatch.setattr(app_module.chat_service, "chat", fake_chat)

    response = client.post(
        "/api/chat",
        json={"user_id": "u1", "session_id": "s1", "message": "你搜一下"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["tool_calls"][0]["name"] == "search_session"
    assert payload["loaded_skills"][0]["name"] == "memory-tools"
    assert payload["activity_traces"][0]["type"] == "skill"
    assert payload["activity_traces"][1]["type"] == "tool"
    assert payload["approval_request"] is None
    assert payload["choice_request"] is None


def test_chat_api_returns_approval_request(monkeypatch) -> None:
    client = TestClient(app_module.app)

    def fake_chat(user_id: str, session_id: str, message: str, llm_user_message: str | None = None):
        _ = user_id
        _ = session_id
        _ = message
        _ = llm_user_message
        return SimpleNamespace(
            reply="等待审批",
            memory_hits=0,
            title="标题",
            tool_calls=[{"name": "bash_command", "input": "rm root/a.txt",
                         "output": "__APPROVAL_REQUIRED__{}"}],
            loaded_skills=[],
            activity_traces=[],
            approval_request={
                "request_id": "abc123",
                "command": "rm root/a.txt",
                "operation": "delete",
                "reason": "operation delete policy",
            },
            choice_request=None,
        )

    monkeypatch.setattr(app_module.chat_service, "chat", fake_chat)

    response = client.post(
        "/api/chat",
        json={"user_id": "u1", "session_id": "s1", "message": "删一下"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["approval_request"]["request_id"] == "abc123"


def test_chat_api_returns_choice_request(monkeypatch) -> None:
    client = TestClient(app_module.app)

    def fake_chat(user_id: str, session_id: str, message: str, llm_user_message: str | None = None):
        _ = user_id
        _ = session_id
        _ = message
        _ = llm_user_message
        return SimpleNamespace(
            reply="我有点拿不准",
            memory_hits=0,
            title="标题",
            tool_calls=[{"name": "ask_human_choice",
                         "input": "q", "output": "__CHOICE_REQUIRED__{}"}],
            loaded_skills=[],
            activity_traces=[],
            approval_request=None,
            choice_request={
                "request_id": "c123",
                "question": "你更喜欢哪种输出",
                "options": ["简洁", "详细"],
                "allow_custom": True,
            },
        )

    monkeypatch.setattr(app_module.chat_service, "chat", fake_chat)

    response = client.post(
        "/api/chat",
        json={"user_id": "u1", "session_id": "s1", "message": "帮我选"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["choice_request"]["request_id"] == "c123"


def test_submit_choice_accepts_custom_input(monkeypatch, tmp_path: Path) -> None:
    app_module.memory_store = MemoryStore(root_dir=tmp_path)
    app_module.memory_store.create_pending_choice(
        "u1",
        "s1",
        {
            "question": "你更喜欢哪种输出",
            "options": ["简洁", "详细"],
            "allow_custom": True,
        },
    )

    captured: dict[str, str] = {}
    replace_call: dict[str, str] = {}

    original_replace_choice_placeholder_output = app_module.memory_store.replace_choice_placeholder_output

    def tracked_replace_choice_placeholder_output(
        user_id: str,
        session_id: str,
        request_id: str,
        output: str,
    ) -> bool:
        replace_call["user_id"] = user_id
        replace_call["session_id"] = session_id
        replace_call["request_id"] = request_id
        replace_call["output"] = output
        return original_replace_choice_placeholder_output(user_id, session_id, request_id, output)

    monkeypatch.setattr(
        app_module.memory_store,
        "replace_choice_placeholder_output",
        tracked_replace_choice_placeholder_output,
    )

    def fake_chat(
        user_id: str,
        session_id: str,
        message: str,
        llm_user_message: str | None = None,
        use_latest_session_memory: bool = False,
        push_event_func=None,
    ):
        captured["user_id"] = user_id
        captured["session_id"] = session_id
        captured["message"] = message
        captured["llm_user_message"] = llm_user_message or ""
        captured["use_latest_session_memory"] = str(use_latest_session_memory)
        captured["has_push_event_func"] = str(callable(push_event_func))
        return SimpleNamespace(
            reply="收到你的选择",
            memory_hits=0,
            title="标题",
            tool_calls=[],
            loaded_skills=[],
            activity_traces=[],
            approval_request=None,
            choice_request=None,
        )

    monkeypatch.setattr(app_module.chat_service, "chat", fake_chat)
    client = TestClient(app_module.app)

    # fetch generated request_id from store file
    choice_file = tmp_path / "u1" / "choices" / "s1.json"
    parsed = yaml.safe_load(choice_file.read_text(encoding="utf-8"))
    request_id = next(iter(parsed.keys()))

    response = client.post(
        "/api/choices/submit",
        json={
            "user_id": "u1",
            "session_id": "s1",
            "request_id": request_id,
            "selected_option": "",
            "custom_input": "我想要图文并茂",
        },
    )

    assert response.status_code == 200
    assert captured["user_id"] == "u1"
    assert captured["session_id"] == "s1"
    assert "我想要图文并茂" in captured["message"]
    assert captured["llm_user_message"] == "针对问题【你更喜欢哪种输出】我的选择是：我想要图文并茂"
    assert captured["use_latest_session_memory"] == "True"
    assert captured["has_push_event_func"] == "True"
    assert replace_call["user_id"] == "u1"
    assert replace_call["session_id"] == "s1"
    assert replace_call["request_id"] == request_id
    assert replace_call["output"] == "用户选择: 我想要图文并茂"


def test_get_conversation_messages_returns_tool_calls(monkeypatch) -> None:
    client = TestClient(app_module.app)

    def fake_get_session_messages(user_id: str, session_id: str):
        _ = user_id
        _ = session_id
        return [
            {
                "ts": "2026-04-08T10:00:00",
                "role": "assistant",
                "content": "已检索",
                "message_id": 2,
                "tool_calls": [
                    {
                        "name": "search_session",
                        "input": "关键字",
                        "output": "#2 ...",
                    }
                ],
                "loaded_skills": [
                    {
                        "name": "memory-tools",
                        "source": "global",
                        "dir": "skills/global/memory-tools",
                    }
                ],
                "activity_traces": [
                    {
                        "type": "skill",
                        "name": "memory-tools",
                        "source": "global",
                        "dir": "skills/global/memory-tools",
                        "input": "",
                        "output": "loaded",
                    },
                    {
                        "type": "tool",
                        "name": "search_session",
                        "source": "",
                        "dir": "",
                        "input": "关键字",
                        "output": "#2 ...",
                    },
                ],
            }
        ]

    monkeypatch.setattr(
        app_module.chat_service,
        "get_session_messages",
        fake_get_session_messages,
    )

    response = client.get("/api/conversations/s1/messages",
                          params={"user_id": "u1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"][0]["tool_calls"][0]["name"] == "search_session"
    assert payload["items"][0]["loaded_skills"][0]["name"] == "memory-tools"
    assert payload["items"][0]["activity_traces"][0]["type"] == "skill"


def test_get_settings_returns_long_term_memory_and_skill_toggles(
    monkeypatch,
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "skills"
    monkeypatch.setattr(app_module, "_skills_root_dir", lambda: skills_root)
    client = TestClient(app_module.app)

    app_module.memory_store = MemoryStore(root_dir=tmp_path)
    app_module.memory_store.set_long_term_memory("u1", "用户偏好中文")

    user_skill_dir = skills_root / "u1"
    user_skill_dir.mkdir(parents=True, exist_ok=True)
    (user_skill_dir / "skills.yaml").write_text(
        yaml.safe_dump(
            {
                "skills": [
                    {"name": "memory-tools", "enabled": False},
                    {"name": "weather", "enabled": True},
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    response = client.get("/api/settings", params={"user_id": "u1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["long_term_memory"] == "用户偏好中文"
    assert payload["skills"][0] == {"name": "memory-tools", "enabled": False}
    assert payload["skills"][1] == {"name": "weather", "enabled": True}
    assert payload["command_policy"]["operations"]["read"] == "allow"
    assert payload["command_policy"]["operations"]["write"] == "ask_human"
    assert payload["command_policy"]["operations"]["delete"] == "ask_human"
    assert payload["command_policy"]["allowed_paths"] == []


def test_put_settings_updates_long_term_memory_and_skill_toggles(
    monkeypatch,
    tmp_path: Path,
) -> None:
    skills_root = tmp_path / "skills"
    monkeypatch.setattr(app_module, "_skills_root_dir", lambda: skills_root)
    client = TestClient(app_module.app)

    app_module.memory_store = MemoryStore(root_dir=tmp_path)

    response = client.put(
        "/api/settings",
        json={
            "user_id": "u2",
            "long_term_memory": "长期记忆\n- 喜欢游泳",
            "skills": [
                {"name": "memory-tools", "enabled": True},
                {"name": "reply-style", "enabled": False},
            ],
            "command_policy": {
                "enabled": True,
                "default_action": "ask_human",
                "operations": {
                    "read": "allow",
                    "write": "ask_human",
                    "delete": "ask_human",
                    "exec": "ask_human",
                },
                "command_allowlist": ["cat", "ls", "rg"],
                "command_blocklist": ["curl"],
                "allowed_paths": ["workdir", ".wozclaw/skills"],
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True

    assert app_module.memory_store.get_long_term_memory("u2") == "长期记忆\n- 喜欢游泳"

    skills_yaml = skills_root / "u2" / "skills.yaml"
    parsed = yaml.safe_load(skills_yaml.read_text(encoding="utf-8"))
    assert parsed == {
        "skills": [
            {"name": "memory-tools", "enabled": True},
            {"name": "reply-style", "enabled": False},
        ]
    }

    command_policy_file = tmp_path / "config" / "u2" / "command_policy.json"
    command_policy = yaml.safe_load(
        command_policy_file.read_text(encoding="utf-8"))
    assert command_policy["operations"]["read"] == "allow"
    assert command_policy["operations"]["write"] == "ask_human"
    assert command_policy["command_allowlist"] == [
        "cat", "ls", "rg"]
    assert command_policy["allowed_paths"] == [
        "workdir", ".wozclaw/skills"]


def test_upload_skill_zip_extracts_and_enables_skill(monkeypatch, tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    monkeypatch.setattr(app_module, "_skills_root_dir", lambda: skills_root)
    client = TestClient(app_module.app)

    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(
            "multi-search-engine/SKILL.md",
            "---\nname: multi-search-engine\ndescription: test\n---\n# demo\n",
        )
        archive.writestr("multi-search-engine/README.md", "demo")

    response = client.post(
        "/api/skills/upload",
        data={"user_id": "u1"},
        files={"file": ("multi-search-engine-2.0.1.zip",
                        buffer.getvalue(), "application/zip")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["name"] == "multi-search-engine"
    assert (skills_root / "u1" / "multi-search-engine" / "SKILL.md").exists()

    parsed = yaml.safe_load(
        (skills_root / "u1" / "skills.yaml").read_text(encoding="utf-8"))
    assert parsed == {"skills": [
        {"name": "multi-search-engine", "enabled": True}]}


def test_upload_skill_zip_falls_back_to_zip_name_without_version(monkeypatch, tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    monkeypatch.setattr(app_module, "_skills_root_dir", lambda: skills_root)
    client = TestClient(app_module.app)

    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(
            "multi-search-engine-2.0.1/SKILL.md",
            "---\ndescription: test\n---\n# demo\n",
        )

    response = client.post(
        "/api/skills/upload",
        data={"user_id": "u1"},
        files={"file": ("multi-search-engine-2.0.1.zip",
                        buffer.getvalue(), "application/zip")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "multi-search-engine"
    assert (skills_root / "u1" / "multi-search-engine" / "SKILL.md").exists()


def test_upload_skill_zip_overwrites_existing_skill(monkeypatch, tmp_path: Path) -> None:
    skills_root = tmp_path / "skills"
    target_dir = skills_root / "u1" / "multi-search-engine"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "SKILL.md").write_text("---\nname: old\n---\nold", encoding="utf-8")
    monkeypatch.setattr(app_module, "_skills_root_dir", lambda: skills_root)
    client = TestClient(app_module.app)

    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(
            "multi-search-engine/SKILL.md",
            "---\nname: multi-search-engine\ndescription: test\n---\n# demo\n",
        )

    response = client.post(
        "/api/skills/upload",
        data={"user_id": "u1"},
        files={"file": ("multi-search-engine.zip",
                        buffer.getvalue(), "application/zip")},
    )

    assert response.status_code == 200
    assert (target_dir / "SKILL.md").read_text(
        encoding="utf-8").startswith("---\nname: multi-search-engine")


def test_get_pending_state_returns_approvals_and_choices(
    tmp_path: Path,
) -> None:
    """Test that /api/sessions/{session_id}/pending-state returns pending requests."""
    memory_store = MemoryStore(root_dir=tmp_path)

    # Create a pending approval
    approval = memory_store.create_pending_approval(
        user_id="u1",
        session_id="s1",
        payload={
            "command": "ls -la",
            "operation": "exec",
            "reason": "test approval",
        },
    )

    # Create a pending choice
    choice = memory_store.create_pending_choice(
        user_id="u1",
        session_id="s1",
        payload={
            "question": "Which option?",
            "options": ["option1", "option2"],
            "allow_custom": True,
        },
    )

    # Mock the app's memory_store
    import types
    app_module.memory_store = memory_store
    client = TestClient(app_module.app)

    response = client.get(
        "/api/sessions/s1/pending-state",
        params={"user_id": "u1"},
    )

    assert response.status_code == 200
    data = response.json()

    # Verify pending approvals
    assert "pending_approvals" in data
    assert len(data["pending_approvals"]) == 1
    assert data["pending_approvals"][0]["command"] == "ls -la"
    assert data["pending_approvals"][0]["request_id"] == approval["request_id"]

    # Verify pending choices
    assert "pending_choices" in data
    assert len(data["pending_choices"]) == 1
    assert data["pending_choices"][0]["question"] == "Which option?"
    assert data["pending_choices"][0]["request_id"] == choice["request_id"]


def test_decide_approval_pushes_tool_event_before_resume(monkeypatch, tmp_path: Path) -> None:
    memory_store = MemoryStore(root_dir=tmp_path)
    approval = memory_store.create_pending_approval(
        user_id="u-approve",
        session_id="s-approve",
        payload={
            "command": "rm root/a.txt",
            "operation": "delete",
            "reason": "test approval",
        },
    )

    chat_service = ChatService(memory_store=memory_store)
    chat_service.remember_pending_approval_trace_id(
        "u-approve",
        "s-approve",
        approval["request_id"],
        "trace-123",
    )

    call_order: list[str] = []
    pushed_events: list[dict[str, object]] = []

    class FakeRuntimeAgent:
        def __init__(self, memory_store: MemoryStore, user_id: str, session_id: str) -> None:
            _ = memory_store
            _ = user_id
            _ = session_id

        def run_bash_command_after_approval(self, command: str) -> str:
            call_order.append(f"run:{command}")
            return "approved output"

    def fake_push_tool_trace_event(session_id: str, event: dict[str, object]) -> None:
        call_order.append("push")
        pushed_events.append({"session_id": session_id, **event})

    def fake_resume_after_approval(**kwargs):
        call_order.append("resume")
        return SimpleNamespace(
            reply="resumed reply",
            memory_hits=0,
            title="resumed title",
            tool_calls=[
                {
                    "name": "bash_command",
                    "input": "rm root/a.txt",
                    "output": "approved output",
                }
            ],
            loaded_skills=[],
            activity_traces=[],
            approval_request=None,
            choice_request=None,
        )

    monkeypatch.setattr(app_module, "memory_store", memory_store)
    monkeypatch.setattr(app_module, "chat_service", chat_service)
    monkeypatch.setattr(
        app_module.chat_service,
        "push_tool_trace_event",
        fake_push_tool_trace_event,
    )
    monkeypatch.setattr(
        app_module.chat_service,
        "resume_after_approval",
        fake_resume_after_approval,
    )
    monkeypatch.setattr("wozclaw.agent.ReActMemoryAgent", FakeRuntimeAgent)

    client = TestClient(app_module.app)
    response = client.post(
        "/api/approvals/decide",
        json={
            "user_id": "u-approve",
            "session_id": "s-approve",
            "request_id": approval["request_id"],
            "approved": True,
        },
    )

    assert response.status_code == 200
    assert call_order[:3] == ["run:rm root/a.txt", "push", "resume"]
    assert pushed_events[0]["trace_id"] == "trace-123"
    assert pushed_events[0]["output"] == "approved output"


def test_get_pending_state_requires_user_id_and_session_id() -> None:
    """Test that /api/sessions/{session_id}/pending-state requires user_id and session_id."""
    client = TestClient(app_module.app)

    # Missing user_id - FastAPI returns 422 for missing query params
    response = client.get("/api/sessions/s1/pending-state")
    assert response.status_code == 422


def test_delete_conversation_api_returns_ok(monkeypatch) -> None:
    client = TestClient(app_module.app)

    def fake_delete_conversation(user_id: str, session_id: str) -> bool:
        assert user_id == "u-del"
        assert session_id == "s-del"
        return True

    monkeypatch.setattr(
        app_module.chat_service,
        "delete_conversation",
        fake_delete_conversation,
    )

    response = client.delete(
        "/api/conversations/s-del",
        params={"user_id": "u-del"},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "deleted": True}


def test_delete_conversation_api_rejects_invalid_ids(monkeypatch) -> None:
    client = TestClient(app_module.app)

    def fake_delete_conversation(user_id: str, session_id: str) -> bool:
        _ = user_id
        _ = session_id
        raise ValueError("user_id and session_id are required")

    monkeypatch.setattr(
        app_module.chat_service,
        "delete_conversation",
        fake_delete_conversation,
    )

    response = client.delete(
        "/api/conversations/s-del",
        params={"user_id": "  "},
    )

    assert response.status_code == 400


def test_rename_conversation_api_returns_title(monkeypatch) -> None:
    client = TestClient(app_module.app)

    def fake_rename_conversation(user_id: str, session_id: str, title: str) -> str:
        assert user_id == "u-rename"
        assert session_id == "s-rename"
        assert title == "新的标题"
        return "新的标题"

    monkeypatch.setattr(
        app_module.chat_service,
        "rename_conversation",
        fake_rename_conversation,
    )

    response = client.patch(
        "/api/conversations/s-rename/title",
        json={"user_id": "u-rename", "title": "新的标题"},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "title": "新的标题"}


def test_rename_conversation_api_rejects_empty_title(monkeypatch) -> None:
    client = TestClient(app_module.app)

    def fake_rename_conversation(user_id: str, session_id: str, title: str) -> str:
        _ = user_id
        _ = session_id
        _ = title
        raise ValueError("title is required")

    monkeypatch.setattr(
        app_module.chat_service,
        "rename_conversation",
        fake_rename_conversation,
    )

    response = client.patch(
        "/api/conversations/s-rename/title",
        json={"user_id": "u-rename", "title": "   "},
    )

    assert response.status_code == 400
