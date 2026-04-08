from pathlib import Path


def test_frontend_does_not_render_memory_hits_hint() -> None:
    html_path = Path("src/agent_memory_demo/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "memory_hits:" not in html


def test_frontend_layout_is_not_centered_container() -> None:
    html_path = Path("src/agent_memory_demo/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "justify-content: center;" not in html
    assert "align-items: center;" not in html


def test_frontend_messages_have_horizontal_margin() -> None:
    html_path = Path("src/agent_memory_demo/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "margin-inline:" in html


def test_frontend_messages_use_readable_sans_font() -> None:
    html_path = Path("src/agent_memory_demo/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "font-family: \"PingFang SC\", \"Hiragino Sans GB\", \"Source Han Sans SC\", \"Microsoft YaHei UI\", sans-serif;" in html


def test_frontend_has_global_ui_scale() -> None:
    html_path = Path("src/agent_memory_demo/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "--ui-scale: 1.15;" in html
    assert "transform: scale(var(--ui-scale));" in html
    assert "width: calc(100vw / var(--ui-scale));" in html
    assert "height: calc(100vh / var(--ui-scale));" in html


def test_frontend_has_tool_call_rendering() -> None:
    html_path = Path("src/agent_memory_demo/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "tool_calls" in html
    assert "工具调用" in html


def test_frontend_tool_calls_are_collapsible_and_above_ai_message() -> None:
    html_path = Path("src/agent_memory_demo/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "createElement('details')" in html
    assert "summary.textContent = `工具调用" in html
    assert "addAssistantMsg" in html


def test_frontend_history_load_uses_persisted_tool_calls() -> None:
    html_path = Path("src/agent_memory_demo/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "addAssistantMsg(item.content, item.tool_calls || []);" in html


def test_frontend_tool_calls_have_visible_separator_between_items() -> None:
    html_path = Path("src/agent_memory_demo/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert "lines.push('--------------------');" in html


def test_frontend_tool_trace_left_aligns_with_assistant_message() -> None:
    html_path = Path("src/agent_memory_demo/static/index.html")
    html = html_path.read_text(encoding="utf-8")

    assert ".assistant-block .msg {" in html
    assert "margin-inline: 0;" in html
