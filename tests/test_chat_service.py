import json
from pathlib import Path

from agent_memory_demo.agent import AgentResponse
from agent_memory_demo.memory_store import MemoryStore
from agent_memory_demo.service import ChatService


class DummyAgent:
    def __init__(self) -> None:
        self.last_context = ""

    def respond(self, user_message: str, memory_context: str) -> str:
        self.last_context = memory_context
        return f"echo:{user_message} | ctx:{'中文输出' in memory_context}"


class DummyTitleGenerator:
    def generate_title(self, user_message: str, assistant_reply: str) -> str:
        _ = assistant_reply
        return f"标题:{user_message[:6]}"


class DummyToolAgent:
    def respond(self, user_message: str, memory_context: str) -> AgentResponse:
        _ = user_message
        _ = memory_context
        return AgentResponse(
            text="tool-answer",
            tool_calls=[
                {
                    "name": "search_daily",
                    "input": "蓝莓",
                    "output": "#1 ...",
                }
            ],
        )


def test_chat_service_persists_messages_and_uses_context(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    store.remember_long_term("u1", "用户偏好中文输出", tags=["preference"])

    service = ChatService(
        memory_store=store,
        agent=DummyAgent(),
        title_generator=DummyTitleGenerator(),
    )

    response = service.chat(user_id="u1", session_id="s1", message="你好")

    assert response.reply.startswith("echo:你好")
    assert response.memory_hits >= 1
    assert response.title == "标题:你好"

    session_file = tmp_path / "u1" / "sessions" / "s1.jsonl"
    assert session_file.exists()
    lines = session_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    log_file = tmp_path / "u1" / "logs" / "llm_dialogue.jsonl"
    assert log_file.exists()
    log_rows = [json.loads(item) for item in log_file.read_text(
        encoding="utf-8").strip().splitlines()]
    assert len(log_rows) == 1
    assert log_rows[0]["session_id"] == "s1"
    assert log_rows[0]["user_message"] == "你好"

    conversations = service.list_conversations("u1")
    assert len(conversations) == 1
    assert conversations[0]["session_id"] == "s1"
    assert conversations[0]["title"] == "标题:你好"

    messages = service.get_session_messages("u1", "s1")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_chat_service_only_injects_recent_three_rounds(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = DummyAgent()
    service = ChatService(memory_store=store, agent=agent,
                          title_generator=DummyTitleGenerator())

    # Prepare 4 full rounds first.
    for idx in range(1, 5):
        store.append_session_message("u2", "s2", "user", f"u{idx}")
        store.append_session_message("u2", "s2", "assistant", f"a{idx}")

    service.chat(user_id="u2", session_id="s2", message="当前问题")

    assert "u1" not in agent.last_context
    assert "a1" not in agent.last_context
    assert "u4" in agent.last_context
    assert "[DAILY]" not in agent.last_context
    assert "今天" not in agent.last_context


def test_chat_service_exposes_tool_calls_for_ui(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    service = ChatService(
        memory_store=store,
        agent=DummyToolAgent(),
        title_generator=DummyTitleGenerator(),
    )

    response = service.chat(user_id="u4", session_id="s4", message="今天关于蓝莓")
    assert response.reply == "tool-answer"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "search_daily"

    messages = service.get_session_messages("u4", "s4")
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["tool_calls"][0]["name"] == "search_daily"


def test_conversation_list_orders_by_latest_chat_time(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    service = ChatService(
        memory_store=store,
        agent=DummyAgent(),
        title_generator=DummyTitleGenerator(),
    )

    service.chat(user_id="u3", session_id="s1", message="第一会话首条")
    service.chat(user_id="u3", session_id="s2", message="第二会话首条")

    conv_file = tmp_path / "u3" / "conversations.json"
    conv_data = json.loads(conv_file.read_text(encoding="utf-8"))
    conv_data["s1"]["updated_at"] = "2026-04-08T10:00:00"
    conv_data["s2"]["updated_at"] = "2026-04-08T10:00:01"
    conv_file.write_text(
        json.dumps(conv_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    before = service.list_conversations("u3")
    assert before[0]["session_id"] == "s2"

    service.chat(user_id="u3", session_id="s1", message="第一会话最新")

    conversations = service.list_conversations("u3")
    assert len(conversations) == 2
    assert conversations[0]["session_id"] == "s1"
    assert conversations[1]["session_id"] == "s2"
