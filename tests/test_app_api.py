from fastapi.testclient import TestClient
from types import SimpleNamespace
from pathlib import Path

import yaml

from wozclaw import app as app_module
from wozclaw.memory_store import MemoryStore


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

    def fake_chat(user_id: str, session_id: str, message: str):
        _ = user_id
        _ = session_id
        _ = message
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
