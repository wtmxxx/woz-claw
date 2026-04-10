from fastapi.testclient import TestClient
from types import SimpleNamespace

from agent_memory_demo import app as app_module


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
