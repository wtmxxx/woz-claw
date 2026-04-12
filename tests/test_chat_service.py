import json
from pathlib import Path

from wozclaw.agent import AgentResponse
from wozclaw.memory_store import MemoryStore
from wozclaw.service import ChatService


class DummyAgent:
    def __init__(self) -> None:
        self.last_context = ""
        self.last_session_memory = ""

    def respond(self, user_message: str, memory_context: str, session_memory: str = "") -> str:
        self.last_context = memory_context
        self.last_session_memory = session_memory
        return f"echo:{user_message} | ctx:{'中文输出' in memory_context}"


class LegacyDummyAgent:
    def __init__(self) -> None:
        self.last_context = ""

    def respond(self, user_message: str, memory_context: str) -> str:
        self.last_context = memory_context
        return f"legacy:{user_message}"


class DummyTitleGenerator:
    def generate_title(self, user_message: str, assistant_reply: str) -> str:
        _ = assistant_reply
        return f"标题:{user_message[:6]}"


class RecordingTitleGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate_title(self, user_message: str, assistant_reply: str) -> str:
        self.calls.append((user_message, assistant_reply))
        return f"标题:{user_message[:6]}"


class DummyToolAgent:
    def respond(self, user_message: str, memory_context: str) -> AgentResponse:
        _ = user_message
        _ = memory_context
        return AgentResponse(
            text="tool-answer",
            loaded_skills=[
                {
                    "name": "memory-tools",
                    "source": "global",
                    "dir": "skills/global/memory-tools",
                }
            ],
            tool_calls=[
                {
                    "name": "search_daily",
                    "input": "蓝莓",
                    "output": "#1 ...",
                }
            ],
            activity_traces=[
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
                    "name": "search_daily",
                    "source": "",
                    "dir": "",
                    "input": "蓝莓",
                    "output": "#1 ...",
                },
            ],
        )


class DummyChoiceAgent:
    def respond(self, user_message: str, memory_context: str) -> AgentResponse:
        _ = user_message
        _ = memory_context
        return AgentResponse(
            text="我有点拿不准",
            tool_calls=[
                {
                    "name": "ask_human_choice",
                    "input": "{\"question\":\"你更想要哪种风格\"}",
                    "output": "__CHOICE_REQUIRED__{\"request_id\":\"c123\",\"question\":\"你更想要哪种风格\",\"options\":[\"简洁\",\"详细\"],\"allow_custom\":true}",
                }
            ],
        )


class DummyApprovalResumeAgent:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def respond(self, user_message: str, memory_context: str) -> AgentResponse:
        _ = memory_context
        self.calls.append(user_message)
        if len(self.calls) > 1:
            return AgentResponse(
                text="已继续完成当前任务",
                tool_calls=[
                    {
                        "name": "search_session",
                        "input": "审批后的上下文",
                        "output": "#2 ...",
                    }
                ],
            )
        return AgentResponse(
            text="检测到需要人工审批，已暂停执行并等待审批结果。",
            tool_calls=[
                {
                    "name": "bash_command",
                    "input": "rm root/a.txt",
                    "output": "__APPROVAL_REQUIRED__{\"request_id\":\"ap1\",\"command\":\"rm root/a.txt\",\"operation\":\"delete\",\"reason\":\"operation delete policy\"}",
                }
            ],
        )


class DummyApprovalTraceAgent:
    def respond(self, user_message: str, memory_context: str) -> AgentResponse:
        _ = user_message
        _ = memory_context
        return AgentResponse(
            text="等待审批",
            tool_calls=[
                {
                    "name": "bash_command",
                    "input": "rm root/a.txt",
                    "output": "__APPROVAL_REQUIRED__{\"request_id\":\"ap-trace\",\"command\":\"rm root/a.txt\"}",
                }
            ],
            activity_traces=[
                {
                    "type": "tool",
                    "name": "bash_command",
                    "source": "",
                    "dir": "",
                    "input": "rm root/a.txt",
                    "output": "__APPROVAL_REQUIRED__{\"request_id\":\"ap-trace\",\"command\":\"rm root/a.txt\"}",
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
    assert not log_file.exists() or not log_file.read_text(encoding="utf-8").strip()

    conversations = service.list_conversations("u1")
    assert len(conversations) == 1
    assert conversations[0]["session_id"] == "s1"
    assert conversations[0]["title"] == "标题:你好"

    messages = service.get_session_messages("u1", "s1")
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_chat_service_does_not_write_llm_dialogue_for_plain_agents(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    store.remember_long_term("u-log", "长期偏好：中文输出")
    store.update_session_state(
        "u-log", "s-log", {"session_memory": "短期摘要：正在排查路径配置"})

    service = ChatService(
        memory_store=store,
        agent=DummyAgent(),
        title_generator=DummyTitleGenerator(),
    )

    service.chat(user_id="u-log", session_id="s-log", message="最近一次提问")

    log_file = tmp_path / "u-log" / "logs" / "llm_dialogue.jsonl"
    assert not log_file.exists() or not log_file.read_text(encoding="utf-8").strip()


def test_chat_service_title_generation_ignores_assistant_reply(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    generator = RecordingTitleGenerator()
    service = ChatService(
        memory_store=store,
        agent=DummyAgent(),
        title_generator=generator,
    )

    service.chat(user_id="u1", session_id="s1", message="第一条用户消息")

    assert len(generator.calls) == 1
    assert generator.calls[0][0] == "第一条用户消息"
    assert generator.calls[0][1] == ""


def test_chat_service_uses_only_long_term_and_session_memory_context(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = DummyAgent()
    service = ChatService(memory_store=store, agent=agent,
                          title_generator=DummyTitleGenerator())

    # Prepare many long messages so token budget excludes older context.
    long_chunk = "x" * 1200
    for idx in range(1, 9):
        store.append_session_message(
            "u2", "s2", "user", f"u{idx}-{long_chunk}")
        store.append_session_message(
            "u2", "s2", "assistant", f"a{idx}-{long_chunk}")

    store.remember_long_term("u2", "长期偏好：中文")
    store.update_session_state("u2", "s2", {"session_memory": "短期摘要：正在排查路径配置"})

    service.chat(user_id="u2", session_id="s2", message="当前问题")

    assert "[SESSION]" not in agent.last_context
    assert "[DAILY]" not in agent.last_context
    assert "u8-" not in agent.last_context
    assert "[LONG_TERM]" in agent.last_context
    assert "长期偏好：中文" in agent.last_context
    assert "[SESSION_MEMORY]" not in agent.last_context
    assert "短期摘要：正在排查路径配置" not in agent.last_context
    assert "短期摘要：正在排查路径配置" in agent.last_session_memory
    assert "user: 当前问题" in agent.last_session_memory


def test_chat_service_legacy_agents_receive_session_memory_via_merged_context(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = LegacyDummyAgent()
    service = ChatService(memory_store=store, agent=agent,
                          title_generator=DummyTitleGenerator())

    store.remember_long_term("u2b", "长期偏好：中文")
    store.update_session_state(
        "u2b", "s2b", {"session_memory": "短期摘要：正在排查路径配置"})

    service.chat(user_id="u2b", session_id="s2b", message="当前问题")

    assert "[SESSION_MEMORY]" in agent.last_context
    assert "短期摘要：正在排查路径配置" in agent.last_context


def test_chat_service_can_use_latest_session_memory_after_appending_user_message(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = DummyAgent()
    service = ChatService(memory_store=store, agent=agent,
                          title_generator=DummyTitleGenerator())

    store.update_session_state("u2c", "s2c", {"session_memory": "已有摘要"})

    service.chat(
        user_id="u2c",
        session_id="s2c",
        message="这是用户选择消息",
        use_latest_session_memory=True,
    )

    assert "user: 这是用户选择消息" in agent.last_session_memory


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
    assert len(response.loaded_skills) == 1
    assert response.loaded_skills[0]["name"] == "memory-tools"
    assert len(response.activity_traces) == 2
    assert response.activity_traces[0]["type"] == "skill"
    assert response.activity_traces[1]["type"] == "tool"

    messages = service.get_session_messages("u4", "s4")
    assert messages[-1]["role"] == "assistant"
    assert messages[-1]["tool_calls"][0]["name"] == "search_daily"
    assert messages[-1]["loaded_skills"][0]["name"] == "memory-tools"
    assert messages[-1]["activity_traces"][0]["type"] == "skill"
    assert messages[-1]["activity_traces"][1]["type"] == "tool"


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


def test_chat_service_exposes_choice_request_for_ui(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    service = ChatService(
        memory_store=store,
        agent=DummyChoiceAgent(),
        title_generator=DummyTitleGenerator(),
    )

    response = service.chat(user_id="u12", session_id="s12", message="给我建议")
    assert response.reply == "我有点拿不准"
    assert response.choice_request is not None
    assert response.choice_request["request_id"] == "c123"
    assert response.choice_request["allow_custom"] is True


def test_chat_service_delete_conversation_delegates_to_store(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    service = ChatService(
        memory_store=store,
        agent=DummyAgent(),
        title_generator=DummyTitleGenerator(),
    )

    store.set_conversation_title("u-del", "s-del", "待删除")
    store.append_session_message("u-del", "s-del", "user", "hello")

    deleted = service.delete_conversation("u-del", "s-del")

    assert deleted is True
    assert service.list_conversations("u-del") == []


def test_chat_service_same_choice_request_id_does_not_duplicate_pending_choice(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    service = ChatService(
        memory_store=store,
        agent=DummyChoiceAgent(),
        title_generator=DummyTitleGenerator(),
    )

    response = service.chat(user_id="u121", session_id="s121", message="给我建议")
    assert response.choice_request is not None
    assert response.choice_request["request_id"] == "c123"

    pending = store.get_pending_choices("u121", "s121")
    assert len(pending) == 1
    assert pending[0]["request_id"] == "c123"

    popped = store.pop_pending_choice("u121", "s121", "c123")
    assert popped is not None
    assert store.get_pending_choices("u121", "s121") == []


def test_chat_service_resume_after_approval_continues_flow(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    service = ChatService(
        memory_store=store,
        agent=DummyApprovalResumeAgent(),
        title_generator=DummyTitleGenerator(),
    )

    first = service.chat(user_id="u20", session_id="s20", message="删除旧文件并继续")
    assert first.approval_request is not None
    assert first.approval_request["request_id"] == "ap1"

    resumed = service.resume_after_approval(
        user_id="u20",
        session_id="s20",
        request_id="ap1",
        command="rm root/a.txt",
        output="ok",
        approved=True,
    )

    assert resumed.reply == "已继续完成当前任务"
    assert resumed.tool_calls[0]["name"] == "bash_command"
    assert resumed.tool_calls[1]["name"] == "search_session"
    assert len(resumed.activity_traces) >= 1
    assert resumed.activity_traces[0]["type"] == "tool"
    assert resumed.activity_traces[0]["name"] == "bash_command"
    assert resumed.activity_traces[0]["output"] == "ok"


def test_resume_after_approval_replaces_placeholder_output_in_history(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    service = ChatService(
        memory_store=store,
        agent=DummyApprovalResumeAgent(),
        title_generator=DummyTitleGenerator(),
    )

    first = service.chat(user_id="u21", session_id="s21", message="删除并继续")
    assert first.approval_request is not None
    assert first.approval_request["request_id"] == "ap1"

    service.resume_after_approval(
        user_id="u21",
        session_id="s21",
        request_id="ap1",
        command="rm root/a.txt",
        output="ok",
        approved=True,
    )

    rows = service.get_session_messages("u21", "s21")
    assistant_rows = [row for row in rows if row["role"] == "assistant"]
    assert len(assistant_rows) >= 1
    tool_rows = [row for row in assistant_rows if row["tool_calls"]]
    assert len(tool_rows) >= 1
    first_assistant_tool_calls = tool_rows[0]["tool_calls"]
    assert first_assistant_tool_calls[0]["name"] == "bash_command"
    assert first_assistant_tool_calls[0]["output"] == "ok"
    first_assistant_activity_traces = tool_rows[0]["activity_traces"]
    assert first_assistant_activity_traces[0]["name"] == "bash_command"
    assert first_assistant_activity_traces[0]["output"] == "ok"


def test_chat_service_does_not_persist_approval_placeholder_tool_trace(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    service = ChatService(
        memory_store=store,
        agent=DummyApprovalResumeAgent(),
        title_generator=DummyTitleGenerator(),
    )

    first = service.chat(user_id="u22", session_id="s22", message="删除并继续")
    assert first.approval_request is not None

    rows = service.get_session_messages("u22", "s22")
    assistant_rows = [row for row in rows if row["role"] == "assistant"]
    assert len(assistant_rows) >= 1
    pending_assistant = assistant_rows[0]
    assert pending_assistant["tool_calls"] == []
    assert pending_assistant["activity_traces"] == []


def test_chat_service_does_not_add_approval_interrupt_message_to_session_memory(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    service = ChatService(
        memory_store=store,
        agent=DummyApprovalResumeAgent(),
        title_generator=DummyTitleGenerator(),
    )

    result = service.chat(user_id="u24", session_id="s24", message="删除并继续")
    assert result.approval_request is not None

    state = store.get_session_state("u24", "s24")
    text = str(state.get("session_memory", ""))
    assert "检测到需要人工审批，已暂停执行并等待审批结果。" not in text
    assert "__APPROVAL_REQUIRED__" not in text


def test_chat_service_filters_approval_placeholder_from_activity_traces(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    service = ChatService(
        memory_store=store,
        agent=DummyApprovalTraceAgent(),
        title_generator=DummyTitleGenerator(),
    )

    result = service.chat(user_id="u23", session_id="s23", message="执行危险命令")
    assert result.approval_request is not None
    assert result.activity_traces == []


def test_resume_after_approval_prefers_in_process_runtime_state(monkeypatch, tmp_path: Path) -> None:
    class RuntimeStateAgent:
        instances: list["RuntimeStateAgent"] = []

        def __init__(self, memory_store: MemoryStore, user_id: str, session_id: str) -> None:
            _ = memory_store
            _ = user_id
            _ = session_id
            self.calls: list[str] = []
            RuntimeStateAgent.instances.append(self)

        def respond(self, user_message: str, memory_context: str) -> AgentResponse:
            _ = memory_context
            self.calls.append(user_message)
            if len(self.calls) == 1:
                return AgentResponse(
                    text="检测到需要人工审批，已暂停执行并等待审批结果。",
                    tool_calls=[
                        {
                            "name": "bash_command",
                            "input": "rm root/a.txt",
                            "output": "__APPROVAL_REQUIRED__{\"request_id\":\"apx\",\"command\":\"rm root/a.txt\",\"operation\":\"delete\",\"reason\":\"operation delete policy\"}",
                        }
                    ],
                )
            return AgentResponse(text="基于同一运行时状态继续完成")

    from wozclaw import service as service_module

    monkeypatch.setattr(service_module, "ReActMemoryAgent", RuntimeStateAgent)

    store = MemoryStore(root_dir=tmp_path)
    chat_service = ChatService(
        memory_store=store, title_generator=DummyTitleGenerator())

    first = chat_service.chat("u30", "s30", "执行并继续")
    assert first.approval_request is not None
    assert first.approval_request["request_id"] == "apx"
    assert len(RuntimeStateAgent.instances) == 1

    resumed = chat_service.resume_after_approval(
        user_id="u30",
        session_id="s30",
        request_id="apx",
        command="rm root/a.txt",
        output="ok",
        approved=True,
    )

    assert resumed.reply == "基于同一运行时状态继续完成"
    assert len(RuntimeStateAgent.instances) == 1
    assert len(RuntimeStateAgent.instances[0].calls) == 2


def test_resume_after_approval_appends_tool_output_to_session_memory(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    service = ChatService(
        memory_store=store,
        agent=DummyApprovalResumeAgent(),
        title_generator=DummyTitleGenerator(),
    )

    first = service.chat(user_id="u25", session_id="s25", message="执行命令")
    assert first.approval_request is not None

    initial_state = store.get_session_state("u25", "s25")
    initial_memory = str(initial_state.get("session_memory", "")).strip()

    resumed = service.resume_after_approval(
        user_id="u25",
        session_id="s25",
        request_id=first.approval_request["request_id"],
        command="rm root/a.txt",
        output="removed successfully",
        approved=True,
    )

    state = store.get_session_state("u25", "s25")
    updated_memory = str(state.get("session_memory", "")).strip()
    assert "[tool] bash_command" in updated_memory
    assert "输入: rm root/a.txt" in updated_memory
    assert "输出: removed successfully" in updated_memory
    assert updated_memory != initial_memory


def test_resume_after_approval_keeps_session_memory_updated(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    store.remember_long_term("u26", "长期记忆")

    class AgentCapturingLLMRequest:
        def __init__(self, memory_store: MemoryStore, user_id: str, session_id: str) -> None:
            _ = memory_store
            _ = user_id
            _ = session_id
            self.calls: list[str] = []

        def respond(self, user_message: str, memory_context: str, session_memory: str = "") -> AgentResponse:
            _ = memory_context
            self.calls.append(user_message)
            if len(self.calls) == 1:
                return AgentResponse(
                    text="需要审批",
                    tool_calls=[
                        {
                            "name": "bash_command",
                            "input": "rm root/a.txt",
                            "output": "__APPROVAL_REQUIRED__{\"request_id\":\"ax\",\"command\":\"rm root/a.txt\"}",
                        }
                    ],
                )
            return AgentResponse(
                text="继续完成"
            )

    agent = AgentCapturingLLMRequest(store, "u26", "s26")
    service = ChatService(
        memory_store=store,
        agent=agent,
        title_generator=DummyTitleGenerator(),
    )

    first = service.chat(user_id="u26", session_id="s26", message="执行")
    assert first.approval_request is not None

    resumed = service.resume_after_approval(
        user_id="u26",
        session_id="s26",
        request_id=first.approval_request["request_id"],
        command="rm root/a.txt",
        output="ok",
        approved=True,
    )

    # Verify agent received 2 calls
    assert len(agent.calls) == 2
    state = store.get_session_state("u26", "s26")
    assert "[tool] bash_command" in str(state.get("session_memory", ""))


def test_resume_after_approval_passes_original_question_to_agent(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    class CaptureApprovalResumeAgent:
        def __init__(self, memory_store: MemoryStore, user_id: str, session_id: str) -> None:
            _ = memory_store
            _ = user_id
            _ = session_id
            self.calls: list[str] = []

        def respond(self, user_message: str, memory_context: str, session_memory: str = "") -> AgentResponse:
            _ = memory_context
            _ = session_memory
            self.calls.append(user_message)
            if len(self.calls) == 1:
                return AgentResponse(
                    text="需要审批",
                    tool_calls=[
                        {
                            "name": "bash_command",
                            "input": "rm root/a.txt",
                            "output": "__APPROVAL_REQUIRED__{\"request_id\":\"apx\",\"command\":\"rm root/a.txt\"}",
                        }
                    ],
                )
            return AgentResponse(text="继续完成")

    agent = CaptureApprovalResumeAgent(store, "u27", "s27")
    service = ChatService(
        memory_store=store,
        agent=agent,
        title_generator=DummyTitleGenerator(),
    )

    first = service.chat(user_id="u27", session_id="s27", message="删除旧文件")
    assert first.approval_request is not None

    resumed = service.resume_after_approval(
        user_id="u27",
        session_id="s27",
        request_id=first.approval_request["request_id"],
        command="rm root/a.txt",
        output="ok",
        approved=True,
    )

    assert resumed.reply == "继续完成"
    assert agent.calls == ["删除旧文件", "删除旧文件"]
