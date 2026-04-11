from datetime import datetime
from pathlib import Path

from wozclaw.memory_store import MemoryStore


def test_append_and_load_recent_context(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    store.append_session_message("u1", "s1", "user", "我喜欢简洁回答")
    store.append_daily_message("u1", "user", "今天讨论 memory demo")
    store.remember_long_term("u1", "用户偏好中文输出")

    result = store.load_context(
        "u1", "s1", query="中文输出", session_limit=5, daily_limit=5, long_term_top_k=3)

    assert len(result.session_messages) == 1
    assert len(result.daily_messages) == 1
    assert len(result.long_term_hits) == 1
    assert "中文" in result.long_term_hits[0]["content"]


def test_files_created_in_expected_layout(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    store.append_session_message("u9", "s9", "user", "hello")
    store.append_daily_message("u9", "assistant", "hi")
    store.remember_long_term("u9", "长期记忆")

    user_root = tmp_path / "u9"
    assert (user_root / "sessions" / "s9.jsonl").exists()
    assert (user_root / "memory.md").exists()

    daily_dir = user_root / "daily"
    files = list(daily_dir.glob("*.jsonl"))
    assert len(files) == 1


def test_long_term_memory_maintenance_without_ids_or_tags(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    store.remember_long_term(
        "u2",
        "用户是一个感性的人\n用户喜欢吃蓝莓、人参果、姑娘果，喜欢游泳，目前正在找工作",
    )
    store.remember_long_term(
        "u2",
        "用户是一个感性的人，喜欢吃人参果、姑娘果，喜欢游泳，最近不喜欢吃蓝莓了",
    )

    long_term_text = store.get_long_term_memory("u2")
    assert "蓝莓、人参果、姑娘果" not in long_term_text
    assert "用户是一个感性的人" in long_term_text
    assert "最近不喜欢吃蓝莓了" in long_term_text

    store.set_long_term_memory("u2", "长期记忆总览\n- 用户偏好中文\n- 用户偏好简洁")
    updated_text = store.get_long_term_memory("u2")
    assert "长期记忆总览" in updated_text
    assert "用户偏好简洁" in updated_text

    removed = store.remove_long_term_memory("u2", "用户偏好简洁")
    assert removed is True
    final_text = store.get_long_term_memory("u2")
    assert "用户偏好简洁" not in final_text

    assert store.remove_long_term_memory("u2", "不存在内容") is False


def test_conversation_title_and_history_listing(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    store.append_session_message("u3", "s1", "user", "第一轮")
    store.append_session_message("u3", "s1", "assistant", "已收到")
    store.set_conversation_title("u3", "s1", "求职与偏好")

    rows = store.list_conversations("u3")
    assert len(rows) == 1
    assert rows[0]["session_id"] == "s1"
    assert rows[0]["title"] == "求职与偏好"

    history = store.get_session_messages("u3", "s1")
    assert len(history) == 2
    assert history[0]["content"] == "第一轮"


def test_list_conversations_reads_from_conversations_json_only(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    conv_file = tmp_path / "u11" / "conversations.json"
    conv_file.parent.mkdir(parents=True, exist_ok=True)
    conv_file.write_text(
        '{\n'
        '  "s1": {"title": "会话1", "updated_at": "2026-04-08T10:00:00"},\n'
        '  "s2": {"title": "会话2", "updated_at": "2026-04-08T10:00:01"}\n'
        '}\n',
        encoding="utf-8",
    )

    rows = store.list_conversations("u11")
    assert [item["session_id"] for item in rows] == ["s2", "s1"]
    assert rows[0]["title"] == "会话2"


def test_memory_store_search_helpers_for_llm_tools(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    for idx in range(1, 6):
        store.append_session_message("u4", "s4", "user", f"u{idx}")
        store.append_session_message("u4", "s4", "assistant", f"a{idx}")

    store.append_session_message("u4", "s4", "user", "我要找工作")
    store.append_session_message("u4", "s4", "assistant", "你在准备简历")

    hits = store.search_session_messages("u4", "s4", keyword="找工作", limit=3)
    assert len(hits) >= 1
    assert hits[0]["message_id"] == 11
    assert any("找工作" in row["content"] for row in hits)

    session_rows = store.get_session_messages("u4", "s4")
    assert [row["message_id"] for row in session_rows[:4]] == [1, 2, 3, 4]

    window = store.get_session_messages_window(
        "u4",
        "s4",
        message_id=11,
        before=1,
        after=1,
    )
    assert [row["content"] for row in window] == ["a5", "我要找工作", "你在准备简历"]

    store.append_daily_message("u4", "user", "今天的记录")
    daily_hits = store.search_daily_messages("u4", keyword="今天", limit=5)
    assert len(daily_hits) >= 1
    assert daily_hits[0]["message_id"] == 1
    assert any("今天的记录" in row["content"] for row in daily_hits)

    current_day = datetime.now().date().isoformat()
    daily_window = store.get_daily_messages_window(
        user_id="u4",
        message_id=1,
        before=0,
        after=0,
        day=current_day,
    )
    assert len(daily_window) == 1
    assert daily_window[0]["content"] == "今天的记录"


def test_daily_message_window_uses_per_day_anchor_and_legacy_rows(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    day = datetime.now().date().isoformat()
    daily_file = tmp_path / "u10" / "daily" / f"{day}.jsonl"
    daily_file.parent.mkdir(parents=True, exist_ok=True)
    daily_file.write_text(
        f'{{"ts":"{day}T09:00:00","role":"user","content":"早上计划","tags":["daily"],"meta":{{}}}}\n'
        f'{{"ts":"{day}T09:05:00","role":"assistant","content":"确认安排","tags":["daily"],"meta":{{}}}}\n',
        encoding="utf-8",
    )

    rows = store.get_daily_messages_by_date("u10", day)
    assert [row["message_id"] for row in rows] == [1, 2]

    store.append_daily_message("u10", "user", "下午再聊")
    updated_rows = store.get_daily_messages_by_date("u10", day)
    assert [row["message_id"] for row in updated_rows] == [1, 2, 3]

    window = store.get_daily_messages_window(
        user_id="u10",
        message_id=2,
        before=1,
        after=1,
        day=day,
    )
    assert [row["content"] for row in window] == ["早上计划", "确认安排", "下午再聊"]


def test_session_message_ids_are_assigned_for_existing_legacy_rows(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    session_file = tmp_path / "u8" / "sessions" / "s8.jsonl"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text(
        '{"ts":"2026-04-08T10:00:00","role":"user","content":"第一条","tags":["session"],"meta":{}}\n'
        '{"ts":"2026-04-08T10:00:01","role":"assistant","content":"第二条","tags":["session"],"meta":{}}\n',
        encoding="utf-8",
    )

    rows = store.get_session_messages("u8", "s8")
    assert [row["message_id"] for row in rows] == [1, 2]

    store.append_session_message("u8", "s8", "user", "第三条")
    updated_rows = store.get_session_messages("u8", "s8")
    assert [row["message_id"] for row in updated_rows] == [1, 2, 3]
    assert updated_rows[-1]["content"] == "第三条"


def test_search_daily_with_date_and_invalid_date_fallback(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    user_root = tmp_path / "u5" / "daily"
    user_root.mkdir(parents=True, exist_ok=True)

    (user_root / "2026-04-07.jsonl").write_text(
        '{"ts":"2026-04-07T10:00:00","role":"user","content":"昨天找工作计划","tags":["daily"],"meta":{}}\n',
        encoding="utf-8",
    )
    (user_root / "2026-04-08.jsonl").write_text(
        '{"ts":"2026-04-08T10:00:00","role":"user","content":"今天游泳安排","tags":["daily"],"meta":{}}\n',
        encoding="utf-8",
    )

    exact_day_hits = store.search_daily_messages(
        "u5", keyword="找工作", limit=10, day="2026-04-07")
    assert len(exact_day_hits) == 1
    assert "昨天找工作计划" in exact_day_hits[0]["content"]

    invalid_day_hits = store.search_daily_messages(
        "u5", keyword="游泳", limit=10, day="bad-date")
    assert len(invalid_day_hits) == 1
    assert "今天游泳安排" in invalid_day_hits[0]["content"]


def test_load_context_daily_limit_zero_excludes_daily_messages(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    store.append_session_message("u6", "s6", "user", "只看会话")
    store.append_daily_message("u6", "user", "不应进入默认上下文")

    result = store.load_context(
        "u6",
        "s6",
        query="测试",
        session_limit=6,
        daily_limit=0,
    )

    assert len(result.session_messages) == 1
    assert result.daily_messages == []


def test_build_prompt_context_omits_empty_sections(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    context_text = store.build_prompt_context(
        store.load_context(
            "u7",
            "s7",
            query="测试",
            session_limit=0,
            daily_limit=0,
        )
    )

    assert "[SESSION]" not in context_text
    assert "[DAILY]" not in context_text
    assert "[LONG_TERM]" not in context_text


def test_load_context_session_uses_token_budget_not_round_limit(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    # Create several short messages that should all fit in a generous token budget.
    for i in range(5):
        store.append_session_message("u12", "s12", "user", f"短消息{i}")

    result = store.load_context(
        "u12",
        "s12",
        query="测试",
        session_limit=1,
        session_token_budget=200,
        daily_limit=0,
    )

    # New behavior: session selection should follow token budget, not session_limit rounds.
    assert len(result.session_messages) > 1


def test_pending_react_state_is_stored_in_session_memory_file(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    store.set_pending_react_state(
        "u13",
        "s13",
        "req13",
        {"user_message": "继续", "prompt_context": "ctx"},
    )

    session_state_file = tmp_path / "u13" / "session_memory" / "s13.json"
    react_state_file = tmp_path / "u13" / "react_state" / "s13.json"
    assert session_state_file.exists()
    assert not react_state_file.exists()

    popped = store.pop_pending_react_state("u13", "s13", "req13")
    assert popped is not None
    assert popped["user_message"] == "继续"


def test_load_context_includes_session_memory(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    store.update_session_state("u14", "s14", {"session_memory": "这是会话压缩摘要"})

    context = store.load_context(
        "u14",
        "s14",
        query="测试",
        session_limit=0,
        daily_limit=0,
    )
    prompt = store.build_prompt_context(context)

    assert "[SESSION_MEMORY]" in prompt
    assert "这是会话压缩摘要" in prompt


def test_load_context_does_not_truncate_session_memory_by_token_budget(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    full_session_memory = "\n".join(
        [f"line-{i}-这是很长的会话记忆内容" for i in range(1, 21)]
    )
    store.update_session_state(
        "u14b",
        "s14b",
        {"session_memory": full_session_memory},
    )

    context = store.load_context(
        "u14b",
        "s14b",
        query="测试",
        session_limit=0,
        session_token_budget=10,
        daily_limit=0,
    )

    assert context.session_memory == full_session_memory


def test_session_memory_accumulates_dialogue_and_tool_operations(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    store.append_session_message("u15", "s15", "user", "你好")
    store.append_session_message(
        "u15",
        "s15",
        "assistant",
        "我先检查文件",
    )
    store.append_session_memory_tool_trace(
        "u15", "s15", "bash_command", "ls root/", "README.md"
    )

    state = store.get_session_state("u15", "s15")
    text = str(state.get("session_memory", ""))
    assert "user: 你好" in text
    assert "assistant: 我先检查文件" in text
    assert "[tool] bash_command" in text
    assert "输入: ls root/" in text
    assert "输出: README.md" in text


def test_session_memory_records_assistant_tool_prelude_before_tool_trace(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    store.append_session_memory_message(
        "u15a",
        "s15a",
        "assistant",
        "我来查看一下当前的工作目录。",
    )
    store.append_session_memory_tool_trace(
        "u15a",
        "s15a",
        "bash_command",
        "pwd",
        "/workdir/demo",
    )

    state = store.get_session_state("u15a", "s15a")
    text = str(state.get("session_memory", ""))
    assert text.index("assistant: 我来查看一下当前的工作目录。") < text.index(
        "[tool] bash_command")


def test_session_memory_ignores_approval_placeholder_tool_output(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    store.append_session_message("u16", "s16", "assistant", "等待审批")
    store.append_session_memory_tool_trace(
        "u16",
        "s16",
        "bash_command",
        "rm root/a.txt",
        '__APPROVAL_REQUIRED__{"request_id":"ap16","command":"rm root/a.txt"}',
    )

    state = store.get_session_state("u16", "s16")
    text = str(state.get("session_memory", ""))
    assert "等待审批" in text
    assert "__APPROVAL_REQUIRED__" not in text
    assert "[tool] bash_command" not in text


def test_session_memory_ignores_approval_interrupt_error_text(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    store.append_session_message("u16aa", "s16aa", "assistant", "等待审批")
    store.append_session_memory_tool_trace(
        "u16aa",
        "s16aa",
        "bash_command",
        "rm root/a.txt",
        "Error: 检测到需要人工审批，已暂停执行并等待审批结果。",
    )

    state = store.get_session_state("u16aa", "s16aa")
    text = str(state.get("session_memory", ""))
    assert "等待审批" in text
    assert "Error: 检测到需要人工审批，已暂停执行并等待审批结果。" not in text
    assert "[tool] bash_command" not in text


def test_session_memory_ignores_choice_interrupt_error_text(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    store.append_session_message("u16ab", "s16ab", "assistant", "等待用户选择")
    store.append_session_memory_tool_trace(
        "u16ab",
        "s16ab",
        "ask_human_choice",
        '{"question":"你更喜欢哪种风格"}',
        "Error: 检测到需要做出选择，已暂停执行并等待用户选择。",
    )

    state = store.get_session_state("u16ab", "s16ab")
    text = str(state.get("session_memory", ""))
    assert "等待用户选择" in text
    assert "Error: 检测到需要做出选择，已暂停执行并等待用户选择。" not in text
    assert "[tool] ask_human_choice" not in text


def test_session_memory_keeps_choice_input_but_drops_choice_required_output(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    store.append_session_memory_tool_trace(
        "u16ac",
        "s16ac",
        "ask_human_choice",
        '{"question":"请选择要查询哪个城市的天气：","options":"上海,襄阳","allow_custom":true}',
        '__CHOICE_REQUIRED__{"request_id":"11d2ed158e2f","question":"请选择要查询哪个城市的天气：","options":["上海","襄阳"],"allow_custom":true}',
    )

    state = store.get_session_state("u16ac", "s16ac")
    text = str(state.get("session_memory", ""))
    assert "[tool] ask_human_choice" in text
    assert "输入:" in text
    assert "__CHOICE_REQUIRED__" not in text


def test_session_memory_keeps_compact_context_output_only(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    store.append_session_message("u16b", "s16b", "assistant", "已压缩上下文")
    store.append_session_memory_tool_trace(
        "u16b",
        "s16b",
        "compact_context",
        '{"summary": "用户询问当前工作目录"}',
        "session memory updated",
    )

    state = store.get_session_state("u16b", "s16b")
    text = str(state.get("session_memory", ""))
    assert "已压缩上下文" in text
    assert "[tool] compact_context" in text
    assert '输入: {"summary": "用户询问当前工作目录"}' not in text
    assert "输出: session memory updated" in text


def test_session_memory_keeps_unrestricted_tool_even_with_placeholder_prefix(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    store.append_session_memory_tool_trace(
        "u16bx",
        "s16bx",
        "custom_tool",
        '{"foo":"bar"}',
        '__APPROVAL_REQUIRED__{"request_id":"x1"}',
    )

    state = store.get_session_state("u16bx", "s16bx")
    text = str(state.get("session_memory", ""))
    assert "[tool] custom_tool" in text
    assert '输入: {"foo":"bar"}' in text
    assert '__APPROVAL_REQUIRED__{"request_id":"x1"}' in text


def test_session_memory_deduplicates_consecutive_same_tool_trace(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    store.append_session_memory_tool_trace(
        "u16c",
        "s16c",
        "bash_command",
        "pwd",
        "/workdir/demo",
    )
    store.append_session_memory_tool_trace(
        "u16c",
        "s16c",
        "bash_command",
        "pwd",
        "/workdir/demo",
    )

    state = store.get_session_state("u16c", "s16c")
    text = str(state.get("session_memory", ""))
    assert text.count("[tool] bash_command") == 1


def test_pending_choice_uses_request_id_as_storage_key(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    item = store.create_pending_choice(
        "u17",
        "s17",
        {
            "request_id": "c17",
            "question": "选哪个",
            "options": ["A", "B"],
        },
    )
    assert item["request_id"] == "c17"

    popped = store.pop_pending_choice("u17", "s17", "c17")
    assert popped is not None
    assert popped["request_id"] == "c17"


def test_pending_choice_duplicate_request_id_overwrites_in_place(tmp_path: Path) -> None:
    store = MemoryStore(root_dir=tmp_path)

    store.create_pending_choice(
        "u18",
        "s18",
        {
            "request_id": "c18",
            "question": "第一次问题",
            "options": ["A"],
        },
    )
    store.create_pending_choice(
        "u18",
        "s18",
        {
            "request_id": "c18",
            "question": "第二次问题",
            "options": ["B"],
        },
    )

    pending = store.get_pending_choices("u18", "s18")
    assert len(pending) == 1
    assert pending[0]["request_id"] == "c18"
    assert pending[0]["question"] == "第二次问题"
