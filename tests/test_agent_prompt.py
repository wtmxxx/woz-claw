import asyncio

from wozclaw.agent import LLMDialogueRecorder, ReActMemoryAgent, ApprovalInterrupt
from wozclaw.memory_store import MemoryStore
from pathlib import Path
from types import SimpleNamespace
import subprocess
import yaml


def test_memory_prompt_requires_full_memory_rewrite(tmp_path) -> None:
    agent = ReActMemoryAgent(memory_store=MemoryStore(
        root_dir=tmp_path), user_id="u1", session_id="s1")

    prompt = agent.build_system_prompt("[LONG_TERM]\n用户喜欢游泳")

    assert "完整" in prompt
    assert "你应主动调用 compact_context" in prompt


def test_agent_module_no_langchain_or_langgraph_imports() -> None:
    source = Path("src/wozclaw/agent.py").read_text(encoding="utf-8")

    assert "from langchain_core" not in source
    assert "from langgraph" not in source
    assert "from langchain_openai" not in source


def test_llm_dialogue_recorder_writes_assistant_prelude_before_tool_trace(tmp_path: Path) -> None:
    class DummyModel:
        async def __call__(self, prompt, **kwargs):
            _ = prompt
            _ = kwargs
            return SimpleNamespace(
                content=[
                    {"type": "text", "text": "我来查看一下当前的工作目录。"},
                    {"type": "tool_use", "name": "bash_command",
                        "input": {"command": "pwd"}},
                ]
            )

    store = MemoryStore(root_dir=tmp_path)
    recorder = LLMDialogueRecorder(
        DummyModel(),
        store,
        "u-rec",
        "s-rec",
    )

    response = asyncio.run(
        recorder([SimpleNamespace(role="user", content="查看目录")]))

    assert response.content[0]["text"] == "我来查看一下当前的工作目录。"
    store.append_session_memory_tool_trace(
        "u-rec",
        "s-rec",
        "bash_command",
        "pwd",
        "/workdir/demo",
    )

    state = store.get_session_state("u-rec", "s-rec")
    text = str(state.get("session_memory", ""))
    assert "assistant: 我来查看一下当前的工作目录。" in text
    assert "[tool] bash_command" in text
    assert text.index("assistant: 我来查看一下当前的工作目录。") < text.index(
        "[tool] bash_command")


def test_llm_dialogue_recorder_filters_react_messages_to_system_and_session_memory_only(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class DummyModel:
        async def __call__(self, prompt, **kwargs):
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            return SimpleNamespace(content=[{"type": "text", "text": "ok"}])

    store = MemoryStore(root_dir=tmp_path)
    recorder = LLMDialogueRecorder(
        DummyModel(),
        store,
        "u-filter",
        "s-filter",
    )

    prompt = [
        {"role": "system", "name": "system", "content": [
            {"type": "text", "text": "sys"}]},
        {"role": "user", "name": "user", "content": [
            {"type": "text", "text": "old user"}]},
        {"role": "assistant", "name": "memory-assistant",
            "content": [{"type": "text", "text": "tool thinking"}]},
        {"role": "tool", "name": "bash_command",
            "content": "cwd=/cygdrive/g/workdir"},
        {"role": "user", "name": "session_memory", "content": [
            {"type": "text", "text": "short memory"}]},
        {"role": "user", "name": "user", "content": [
            {"type": "text", "text": "latest user"}]},
    ]

    asyncio.run(recorder(prompt, tools=[{"type": "function"}]))

    assert isinstance(captured["prompt"], list)
    filtered = captured["prompt"]
    assert len(filtered) == 2
    assert filtered[0]["role"] == "system"
    assert filtered[1]["role"] == "user"
    assert filtered[1]["name"] == "session_memory"


def test_llm_dialogue_recorder_does_not_persist_tool_role_output_into_session_memory(tmp_path: Path) -> None:
    class DummyModel:
        async def __call__(self, prompt, **kwargs):
            _ = prompt
            _ = kwargs
            return SimpleNamespace(content=[{"type": "text", "text": "ok"}])

    store = MemoryStore(root_dir=tmp_path)
    recorder = LLMDialogueRecorder(
        DummyModel(),
        store,
        "u-tool",
        "s-tool",
    )

    prompt = [
        {"role": "system", "name": "system", "content": [
            {"type": "text", "text": "sys"}]},
        {"role": "tool", "name": "bash_command", "content": "cwd=/workdir/demo"},
        {"role": "user", "name": "session_memory",
            "content": [{"type": "text", "text": "m"}]},
    ]

    asyncio.run(recorder(prompt))

    state = store.get_session_state("u-tool", "s-tool")
    text = str(state.get("session_memory", ""))
    assert "[tool] bash_command" not in text
    assert "cwd=/workdir/demo" not in text


def test_llm_dialogue_recorder_injects_latest_session_memory_for_each_react_node(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class DummyModel:
        async def __call__(self, prompt, **kwargs):
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            return SimpleNamespace(content=[{"type": "text", "text": "ok"}])

    store = MemoryStore(root_dir=tmp_path)
    store.append_session_memory_tool_trace(
        "u-node-sync",
        "s-node-sync",
        "search_daily",
        '{"keyword":"蓝莓"}',
        "#1 ...",
    )

    recorder = LLMDialogueRecorder(
        DummyModel(),
        store,
        "u-node-sync",
        "s-node-sync",
    )

    prompt = [
        {"role": "system", "name": "system", "content": [
            {"type": "text", "text": "sys"}]},
        {"role": "user", "name": "user", "content": [
            {"type": "text", "text": "latest user"}]},
    ]

    asyncio.run(recorder(prompt, tools=[{"type": "function"}]))

    assert isinstance(captured["prompt"], list)
    filtered = captured["prompt"]
    assert len(filtered) == 2
    assert filtered[0]["role"] == "system"
    assert filtered[1]["role"] == "user"
    assert filtered[1]["name"] == "session_memory"
    content = filtered[1]["content"]
    assert isinstance(content, list)
    assert "[tool] search_daily" in str(content[0].get("text", ""))


def test_llm_dialogue_recorder_auto_compacts_session_memory_before_react_node(tmp_path: Path) -> None:
    class DummyModel:
        async def __call__(self, prompt, **kwargs):
            _ = prompt
            _ = kwargs
            return SimpleNamespace(content=[{"type": "text", "text": "ok"}])

    class DummyCompactModel:
        async def __call__(self, prompt, **kwargs):
            _ = prompt
            _ = kwargs
            return SimpleNamespace(content=[{"type": "text", "text": "压缩后的会话记忆"}])

    store = MemoryStore(root_dir=tmp_path)
    # ASCII text token estimate is roughly len/4, so this is > 30k tokens.
    store.update_session_state(
        "u-compact",
        "s-compact",
        {"session_memory": "a" * 124000},
    )

    recorder = LLMDialogueRecorder(
        DummyModel(),
        store,
        "u-compact",
        "s-compact",
        compact_model=DummyCompactModel(),
        compact_threshold_tokens=30000,
    )

    prompt = [
        {"role": "system", "name": "system", "content": [
            {"type": "text", "text": "sys"}]},
        {"role": "user", "name": "user", "content": [
            {"type": "text", "text": "继续"}]},
    ]

    asyncio.run(recorder(prompt))

    state = store.get_session_state("u-compact", "s-compact")
    text = str(state.get("session_memory", ""))
    assert text.startswith("压缩后的会话记忆")
    assert "assistant: 上下文压缩成功。" not in text
    assert "[tool] compact_context" in text
    assert "输出: 上下文压缩成功" in text


def test_respond_handles_read_only_sys_prompt(monkeypatch, tmp_path) -> None:
    class DummyReadOnlyPromptAgent:
        @property
        def sys_prompt(self) -> str:
            return "readonly"

        async def reply(self, msg):  # noqa: ANN001
            _ = msg
            return SimpleNamespace(content="ok")

    agent = ReActMemoryAgent(memory_store=MemoryStore(
        root_dir=tmp_path), user_id="u1", session_id="s1")
    agent._agent = DummyReadOnlyPromptAgent()  # type: ignore[assignment]

    result = agent.respond("你好", "上下文")

    assert result.text == "ok"


def test_respond_returns_immediately_on_approval_interrupt(tmp_path) -> None:
    class DummyApprovalAgent:
        async def reply(self, msg):  # noqa: ANN001
            _ = msg
            raise ApprovalInterrupt("需要人工审批后继续")

    agent = ReActMemoryAgent(memory_store=MemoryStore(
        root_dir=tmp_path), user_id="u-approval", session_id="s-approval")
    agent._agent = DummyApprovalAgent()  # type: ignore[assignment]

    result = agent.respond("执行危险命令", "上下文")

    assert result.text == "需要人工审批后继续"


def test_respond_includes_auto_compact_trace_in_activity_traces(tmp_path: Path) -> None:
    class DummyAgent:
        async def reply(self, msg):  # noqa: ANN001
            _ = msg
            return SimpleNamespace(content="ok")

    class DummyRecorder:
        def __init__(self) -> None:
            self._calls = 0
            self._rows = [
                {
                    "name": "compact_context",
                    "input": "",
                    "output": "上下文压缩成功",
                }
            ]

        def consume_auto_tool_traces(self):
            self._calls += 1
            if self._calls == 1:
                return []
            rows = list(self._rows)
            self._rows = []
            return rows

    agent = ReActMemoryAgent(memory_store=MemoryStore(
        root_dir=tmp_path), user_id="u-auto-trace", session_id="s-auto-trace")
    agent._agent = DummyAgent()  # type: ignore[assignment]
    agent._dialogue_recorder = DummyRecorder()  # type: ignore[assignment]

    result = agent.respond("继续", "上下文")

    assert result.text == "ok"
    assert any(item.get("name") == "compact_context"
               for item in result.tool_calls)
    assert any(
        item.get("type") == "tool" and item.get("name") == "compact_context"
        for item in result.activity_traces
    )


def test_respond_cancels_running_reply_when_approval_flag_set(tmp_path) -> None:
    class DummySlowAgent:
        def __init__(self, owner: ReActMemoryAgent) -> None:
            self.owner = owner

        async def reply(self, msg):  # noqa: ANN001
            _ = msg
            self.owner._approval_interrupt_requested = True
            await asyncio.sleep(1)
            return SimpleNamespace(content="should-not-reach")

    agent = ReActMemoryAgent(memory_store=MemoryStore(
        root_dir=tmp_path), user_id="u-cancel", session_id="s-cancel")
    agent._agent = DummySlowAgent(agent)  # type: ignore[assignment]

    result = agent.respond("执行危险命令", "上下文")

    assert "人工审批" in result.text


def test_respond_sends_session_memory_as_user_message(monkeypatch, tmp_path) -> None:
    captured = {}

    class DummyAgent:
        async def reply(self, msg):  # noqa: ANN001
            captured["msg"] = msg
            return SimpleNamespace(content="ok")

    agent = ReActMemoryAgent(memory_store=MemoryStore(
        root_dir=tmp_path), user_id="u-msg", session_id="s-msg")
    agent._agent = DummyAgent()  # type: ignore[assignment]

    result = agent.respond("最近问题", "[LONG_TERM]\n长期记忆", session_memory="会话记忆")

    assert result.text == "ok"
    assert isinstance(captured["msg"], list)
    assert len(captured["msg"]) == 2
    assert captured["msg"][0].role == "user"
    assert captured["msg"][0].name == "session_memory"
    assert captured["msg"][0].content == "会话记忆"
    assert captured["msg"][1].role == "user"
    assert captured["msg"][1].name == "user"
    assert captured["msg"][1].content == "最近问题"


def test_record_tool_trace_updates_session_memory_per_react_node(tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u-node", session_id="s-node")

    agent._record_tool_trace("search_daily", {"keyword": "蓝莓"}, "#1 ...")

    state = store.get_session_state("u-node", "s-node")
    text = str(state.get("session_memory", ""))
    assert "[tool] search_daily" in text
    assert "输入:" in text
    assert "输出: #1 ..." in text


def test_record_tool_trace_respects_session_memory_filters(tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u-node2", session_id="s-node2")

    agent._record_tool_trace(
        "compact_context", '{"summary":"x"}', "上下文压缩成功")
    agent._record_tool_trace(
        "bash_command",
        "rm root/a.txt",
        '__APPROVAL_REQUIRED__{"request_id":"ap1","command":"rm root/a.txt"}',
    )

    state = store.get_session_state("u-node2", "s-node2")
    text = str(state.get("session_memory", ""))
    assert "[tool] compact_context" in text
    assert '输入: {"summary":"x"}' not in text
    assert "输出: 上下文压缩成功" in text
    assert "__APPROVAL_REQUIRED__" not in text


def test_record_tool_trace_includes_ask_human_choice(tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u-choice", session_id="s-choice")

    agent._record_tool_trace(
        "ask_human_choice",
        '{"question":"你更喜欢哪种风格"}',
        '__CHOICE_REQUIRED__{"request_id":"c1","question":"你更喜欢哪种风格"}',
    )

    state = store.get_session_state("u-choice", "s-choice")
    text = str(state.get("session_memory", ""))
    assert "[tool] ask_human_choice" in text
    assert "输入:" in text
    assert "__CHOICE_REQUIRED__" not in text


def test_refresh_prompt_with_latest_session_memory_updates_agent_prompt(tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u-refresh", session_id="s-refresh"
    )

    initial_prompt = "初始提示词"
    agent._active_memory_context = initial_prompt
    agent._active_session_memory = ""
    agent._set_runtime_prompt("初始系统提示词")

    store.append_session_memory_tool_trace(
        "u-refresh", "s-refresh", "search_daily", "蓝莓", "#1 ..."
    )

    agent._refresh_prompt_with_latest_session_memory()

    assert agent._active_session_memory != ""
    assert "[tool] search_daily" in agent._active_session_memory


def test_user_skills_yaml_controls_enabled_skills(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(memory_store=store, user_id="u1", session_id="s1")

    skills_root = tmp_path / "skills"
    enabled_dir = skills_root / "u1" / "memory-tools"
    disabled_dir = skills_root / "u1" / "planner"
    enabled_dir.mkdir(parents=True, exist_ok=True)
    disabled_dir.mkdir(parents=True, exist_ok=True)
    (enabled_dir / "SKILL.md").write_text(
        "---\nname: memory-tools\ndescription: test\n---\n",
        encoding="utf-8",
    )
    (disabled_dir / "SKILL.md").write_text(
        "---\nname: planner\ndescription: test\n---\n",
        encoding="utf-8",
    )

    user_dir = skills_root / "u1"
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "skills.yaml").write_text(
        yaml.safe_dump(
            {
                "skills": [
                    {"name": "memory-tools", "enabled": True},
                    {"name": "planner", "enabled": False},
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(agent, "_skills_root_dir", lambda: skills_root)

    resolved = agent._resolve_enabled_skill_dirs()

    assert resolved == [enabled_dir]


def test_invalid_user_skills_yaml_does_not_crash(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(memory_store=store, user_id="u2", session_id="s2")

    skills_root = tmp_path / "skills"
    skills_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(agent, "_skills_root_dir", lambda: skills_root)

    user_dir = skills_root / "u2"
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "skills.yaml").write_text("skills: [", encoding="utf-8")

    resolved = agent._resolve_enabled_skill_dirs()

    assert resolved == []


def test_register_user_skills_calls_toolkit(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(memory_store=store, user_id="u3", session_id="s3")

    skill_dir = tmp_path / "skills-root" / "memory-tools"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: memory-tools\ndescription: test\n---\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        agent,
        "_resolve_enabled_skills",
        lambda: [{"name": "memory-tools",
                  "source": "global", "dir": str(skill_dir)}],
    )

    class DummyToolkit:
        def __init__(self) -> None:
            self.registered: list[str] = []

        def register_agent_skill(self, path: str) -> None:
            self.registered.append(path)

    toolkit = DummyToolkit()
    agent._register_user_skills(toolkit)  # type: ignore[arg-type]

    assert toolkit.registered == [str(skill_dir)]


def test_toolkit_skill_prompt_uses_xml_and_bash_path(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(memory_store=store, user_id="u3", session_id="s3")

    skill_dir = tmp_path / "skills-root" / "memory-tools"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: memory-tools\ndescription: test skill\n---\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        agent,
        "_resolve_enabled_skills",
        lambda: [{"name": "memory-tools",
                  "source": "global", "dir": str(skill_dir)}],
    )

    toolkit = agent._build_toolkit()
    prompt = toolkit.get_agent_skill_prompt()

    assert prompt is not None
    assert "<SKILLS>" in prompt
    assert "<SKILL>" in prompt
    assert "# Agent Skills" not in prompt
    assert "multiple files" in prompt
    assert "references/" in prompt
    assert "assets/" in prompt
    assert f'{agent._to_bash_path(skill_dir)}/SKILL.md' in prompt


def test_global_and_user_skills_are_merged(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(memory_store=store, user_id="u4", session_id="s4")

    skills_root = tmp_path / "skills"
    global_dir = skills_root / "global" / "global-memory"
    user_dir_skill = skills_root / "u4" / "user-memory"
    global_dir.mkdir(parents=True, exist_ok=True)
    user_dir_skill.mkdir(parents=True, exist_ok=True)
    (global_dir / "SKILL.md").write_text("---\nname: global-memory\ndescription: test\n---\n", encoding="utf-8")
    (user_dir_skill / "SKILL.md").write_text(
        "---\nname: user-memory\ndescription: test\n---\n", encoding="utf-8")

    global_cfg = skills_root / "global" / "skills.yaml"
    global_cfg.write_text(
        yaml.safe_dump({"skills": [
                       {"name": "global-memory", "enabled": True}]}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    user_cfg_dir = skills_root / "u4"
    user_cfg_dir.mkdir(parents=True, exist_ok=True)
    (user_cfg_dir / "skills.yaml").write_text(
        yaml.safe_dump({"skills": [
                       {"name": "user-memory", "enabled": True}]}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(agent, "_skills_root_dir", lambda: skills_root)

    resolved = agent._resolve_enabled_skill_dirs()

    assert resolved == [global_dir, user_dir_skill]


def test_user_setting_overrides_global_skill_toggle(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(memory_store=store, user_id="u5", session_id="s5")

    skills_root = tmp_path / "skills"
    skill_dir = skills_root / "global" / "memory-tools"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text("---\nname: memory-tools\ndescription: test\n---\n", encoding="utf-8")

    global_cfg = skills_root / "global" / "skills.yaml"
    global_cfg.write_text(
        yaml.safe_dump({"skills": [
                       {"name": "memory-tools", "enabled": True}]}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    user_cfg_dir = skills_root / "u5"
    user_cfg_dir.mkdir(parents=True, exist_ok=True)
    (user_cfg_dir / "skills.yaml").write_text(
        yaml.safe_dump({"skills": [
                       {"name": "memory-tools", "enabled": False}]}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(agent, "_skills_root_dir", lambda: skills_root)

    resolved = agent._resolve_enabled_skill_dirs()

    assert resolved == []


def test_user_setting_can_enable_global_skill_without_user_dir(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(memory_store=store, user_id="u9", session_id="s9")

    skills_root = tmp_path / "skills"
    global_dir = skills_root / "global" / "memory-tools"
    global_dir.mkdir(parents=True, exist_ok=True)
    (global_dir / "SKILL.md").write_text(
        "---\nname: memory-tools\ndescription: test\n---\n",
        encoding="utf-8",
    )

    global_cfg = skills_root / "global" / "skills.yaml"
    global_cfg.write_text(
        yaml.safe_dump({"skills": [
                       {"name": "memory-tools", "enabled": True}]}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    user_cfg_dir = skills_root / "u9"
    user_cfg_dir.mkdir(parents=True, exist_ok=True)
    (user_cfg_dir / "skills.yaml").write_text(
        yaml.safe_dump({"skills": [
                       {"name": "memory-tools", "enabled": True}]}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    monkeypatch.setattr(agent, "_skills_root_dir", lambda: skills_root)

    resolved = agent._resolve_enabled_skill_dirs()

    assert resolved == [global_dir]


def test_bash_policy_default_allows_read_but_requires_approval_for_write_and_delete(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u10", session_id="s10")

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        _ = kwargs
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    read_out = agent._run_bash_command("cat root/README.md")
    assert read_out == "ok"

    write_out = agent._run_bash_command("echo hi > root/a.txt")
    assert write_out.startswith("__APPROVAL_REQUIRED__")

    delete_out = agent._run_bash_command("rm root/a.txt")
    assert delete_out.startswith("__APPROVAL_REQUIRED__")


def test_bash_policy_can_deny_when_configured(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    store.set_user_settings(
        "u11",
        {
            "command_policy": {
                "enabled": True,
                "operations": {
                    "read": "allow",
                    "write": "deny",
                    "delete": "ask_human",
                    "exec": "ask_human",
                },
            }
        },
    )
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u11", session_id="s11")

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        _ = kwargs
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    denied = agent._run_bash_command("echo hi > root/a.txt")
    assert denied.startswith("error: command denied by policy")


def test_bash_policy_allowed_paths_bypass_read_write_delete(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    store.set_command_policy(
        "u12",
        {
            "enabled": True,
            "operations": {
                "read": "ask_human",
                "write": "ask_human",
                "delete": "ask_human",
                "exec": "ask_human",
            },
            "allowed_paths": ["root/safe-zone"],
        },
    )
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u12", session_id="s12")

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        _ = kwargs
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    out_read = agent._run_bash_command("cat root/safe-zone/a.txt")
    out_write = agent._run_bash_command("echo hello > root/safe-zone/a.txt")
    out_delete = agent._run_bash_command("rm root/safe-zone/a.txt")

    assert out_read == "ok"
    assert out_write == "ok"
    assert out_delete == "ok"


def test_bash_policy_allowed_paths_do_not_bypass_exec(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    store.set_command_policy(
        "u13",
        {
            "enabled": True,
            "operations": {
                "read": "allow",
                "write": "ask_human",
                "delete": "ask_human",
                "exec": "ask_human",
            },
            "allowed_paths": ["root/safe-zone"],
        },
    )
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u13", session_id="s13")

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        _ = kwargs
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    out_exec = agent._run_bash_command("python root/safe-zone/script.py")
    assert out_exec.startswith("__APPROVAL_REQUIRED__")


def test_bash_command_allows_most_commands(tmp_path) -> None:
    """Test that bash_command now allows find, grep, and other common commands."""
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(memory_store=store, user_id="u6", session_id="s6")

    ok, reason = agent._validate_bash_command("find . -name '*.py'")
    assert ok is True

    ok, reason = agent._validate_bash_command("grep -r test root/")
    assert ok is True

    ok, reason = agent._validate_bash_command("ls -la")
    assert ok is True


def test_bash_command_allows_dotdot_paths(tmp_path) -> None:
    """Test that bash command validation no longer blocks .. traversal tokens."""
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(memory_store=store, user_id="u6", session_id="s6")

    project_root = tmp_path / "project-root"
    (project_root / ".sandbox").mkdir(parents=True, exist_ok=True)
    monkeypatch = __import__("unittest.mock", fromlist=[
                             "MagicMock"]).MagicMock()

    # Paths within workspace should pass now
    ok, reason = agent._validate_bash_command("ls src")
    assert ok is True

    # .. paths are now also allowed
    ok, reason = agent._validate_bash_command("ls ../outside")
    assert ok is True
    assert reason == ""

    # Literal text with .. should not be treated as traversal
    ok, reason = agent._validate_bash_command('echo "a..b"')
    assert ok is True
    assert reason == ""

    # Quoted traversal is treated as literal text
    ok, reason = agent._validate_bash_command('ls "../outside"')
    assert ok is True
    assert reason == ""


def test_bash_path_validation_allows_dotdot(tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u6b", session_id="s6b")

    ok, reason = agent._validate_bash_path("../outside")
    assert ok is True
    assert reason == ""

    ok, reason = agent._validate_bash_path("a..b")
    assert ok is True
    assert reason == ""


def test_bash_command_runs_in_default_sandbox_with_bash(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(memory_store=store, user_id="u7", session_id="s7")

    project_root = tmp_path / "project-root"
    work_dir = project_root / "custom-sandbox"
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(agent, "_project_root_dir", lambda: project_root)
    agent._bash_work_dir = work_dir

    captured: dict[str, object] = {}

    def fake_run(cmd, cwd, capture_output, text, timeout, shell, encoding, errors):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["timeout"] = timeout
        captured["shell"] = shell
        captured["encoding"] = encoding
        captured["errors"] = errors
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = agent._run_bash_command("ls .wozclaw")

    assert result == "ok"
    assert captured["cwd"] == work_dir
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["shell"] is False
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
    assert isinstance(captured["cmd"], list)
    assert captured["cmd"][0] == "bash"
    assert captured["cmd"][1] == "-lc"


def test_bash_command_allows_any_command_without_dotdot(monkeypatch, tmp_path) -> None:
    """Test that commands pass validation as long as they don't contain .."""
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(memory_store=store, user_id="u8", session_id="s8")

    project_root = tmp_path / "project-root"
    configured_sandbox = project_root / "custom-sandbox"
    configured_sandbox.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(agent, "_project_root_dir", lambda: project_root)

    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "sandbox.yaml").write_text(
        "sandbox:\n  writable_dir: custom-sandbox\n",
        encoding="utf-8",
    )

    ok, reason = agent._validate_bash_command("ls custom-sandbox")

    assert ok is True
    assert reason == ""


def test_bash_command_expands_root_and_sandbox_aliases(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(memory_store=store, user_id="u9", session_id="s9")

    expanded = agent._expand_bash_aliases(
        "ls root/src && ls .sandbox"
    )
    assert expanded == "ls root/src && ls .sandbox"


def test_bash_command_expands_dot_sandbox_without_double_replacement(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u9b", session_id="s9b")

    expanded = agent._expand_bash_aliases("ls -la .sandbox/")
    assert expanded == "ls -la .sandbox/"


def test_bash_command_allows_root_path_without_strict_validation(monkeypatch, tmp_path) -> None:
    """Test that root/ paths are allowed without strict path validation."""
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u10", session_id="s10")

    project_root = tmp_path / "project-root"
    work_dir = project_root / "custom-sandbox"
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(agent, "_project_root_dir", lambda: project_root)

    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "sandbox.yaml").write_text(
        "sandbox:\n  writable_dir: custom-sandbox\n",
        encoding="utf-8",
    )

    ok, reason = agent._validate_bash_command("ls -la root/")

    assert ok is True
    assert reason == ""


def test_bash_rewrite_output_paths_converts_work_dir_to_root(tmp_path) -> None:
    """Test that output rewriting is now a no-op."""
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u11", session_id="s11")

    project_root = tmp_path / "project-root"
    work_dir = project_root / "custom-sandbox"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Simulate bash output with absolute work_dir paths
    work_dir_posix = work_dir.resolve().as_posix()
    output = f"Files in {work_dir_posix}/src:\nfile1.py\nfile2.py"

    result = agent._rewrite_output_paths(output, work_dir)

    assert result == output


def test_bash_command_truncates_stdout_to_100k(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u100k", session_id="s100k")

    project_root = tmp_path / "project-root"
    work_dir = project_root / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(agent, "_project_root_dir", lambda: project_root)

    long_stdout = "a" * 120000

    def fake_run(cmd, cwd, capture_output, text, timeout, shell, encoding, errors):  # noqa: ANN001
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=long_stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    output = agent._run_bash_command("echo ok")

    assert len(output) == 100000
    assert output == long_stdout[:100000]


def test_bash_command_strips_ansi_escape_sequences(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u-ansi", session_id="s-ansi")

    project_root = tmp_path / "project-root"
    work_dir = project_root / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(agent, "_project_root_dir", lambda: project_root)

    ansi_output = "\x1b[H\x1b[2J\x1b[3J/cygdrive/g/workdir\n__WOZCLAW_CWD__/cygdrive/g/workdir\n"

    def fake_run(cmd, cwd, capture_output, text, timeout, shell, encoding, errors):  # noqa: ANN001
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=ansi_output, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    output = agent._run_bash_command("pwd")

    assert output == "/cygdrive/g/workdir"
    assert "\x1b" not in output


def test_bash_command_returns_timeout_text_on_timeout(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u-timeout", session_id="s-timeout")

    project_root = tmp_path / "project-root"
    work_dir = project_root / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(agent, "_project_root_dir", lambda: project_root)

    def fake_run(cmd, cwd, capture_output, text, timeout, shell, encoding, errors):  # noqa: ANN001
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(subprocess, "run", fake_run)

    output = agent._run_bash_command(
        "find /cygdrive/g -name '.wozclaw' -type d")

    assert output == "timeout"


def test_bash_supports_pipes_and_redirects(tmp_path) -> None:
    """Test that bash_command now allows pipes and redirects."""
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u12", session_id="s12")

    # These should pass validation now
    ok, reason = agent._validate_bash_command(
        "cat root/file.txt | grep pattern")
    assert ok is True

    ok, reason = agent._validate_bash_command(
        "find . -name '*.py' > root/results.txt")
    assert ok is True

    ok, reason = agent._validate_bash_command("ls -la | wc -l")
    assert ok is True


def test_configured_workdir_handles_relative_paths(monkeypatch, tmp_path) -> None:
    """Test that relative configured workdir is resolved from project root."""
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u13", session_id="s13")

    project_root = tmp_path / "project-root"
    project_root.mkdir(parents=True, exist_ok=True)
    workdir = project_root / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "test.txt").write_text("content", encoding="utf-8")

    monkeypatch.setattr(agent, "_project_root_dir", lambda: project_root)
    agent._bash_work_dir = project_root

    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "path.yaml").write_text(
        "path:\n  workdir: workdir\n",
        encoding="utf-8",
    )

    # Force reload of config by calling the method directly
    configured = agent._configured_work_dir()
    assert configured == workdir

    root_work = agent._root_work_dir()
    assert root_work == workdir


def test_memory_prompt_paths_includes_configured_dirs(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u-path", session_id="s-path")

    project_root = tmp_path / "project-root"
    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "path.yaml").write_text(
        "path:\n  workdir: workdir\n  wozclaw_dir: .wozclaw\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(agent, "_project_root_dir", lambda: project_root)

    prompt = agent.build_system_prompt("ctx")

    workdir = (project_root / "workdir").resolve()
    wozclaw_dir = (project_root / ".wozclaw").resolve()

    assert str(workdir) in prompt or agent._to_bash_path(workdir) in prompt
    assert str(wozclaw_dir) in prompt or agent._to_bash_path(
        wozclaw_dir) in prompt


def test_bash_aliases_preserve_pipes_and_redirects(tmp_path) -> None:
    """Test that command text is preserved when no alias expansion is applied."""
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u14", session_id="s14")

    project_root = tmp_path / "project-root"
    project_root.mkdir(parents=True, exist_ok=True)
    work_dir = project_root / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Mock the paths
    import unittest.mock
    with unittest.mock.patch.object(agent, "_project_root_dir", return_value=project_root):
        with unittest.mock.patch.object(agent, "_root_work_dir", return_value=work_dir):
            # Test pipe preservation
            expanded_pipe = agent._expand_bash_aliases(
                "find root/ -type f | head -20")
            assert "|" in expanded_pipe
            assert "head" in expanded_pipe

            # Test redirection preservation
            expanded_redirect = agent._expand_bash_aliases(
                "ls -la root/ > root/output.txt")
            assert ">" in expanded_redirect
            assert "output.txt" in expanded_redirect

            # Test && operator
            expanded_and = agent._expand_bash_aliases("ls root/a && ls root/b")
            assert "&&" in expanded_and

            assert expanded_pipe == "find root/ -type f | head -20"
            assert expanded_redirect == "ls -la root/ > root/output.txt"
            assert expanded_and == "ls root/a && ls root/b"


def test_nested_root_prefix_not_double_replaced(tmp_path) -> None:
    """Test that root/root text is left untouched without alias rewriting."""
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u15", session_id="s15")

    project_root = tmp_path / "project-root"
    work_dir = project_root / "workdir"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Create a root/ subdirectory inside workdir to verify it's not double-replaced
    root_subdir = work_dir / "root"
    root_subdir.mkdir(parents=True, exist_ok=True)

    import unittest.mock
    with unittest.mock.patch.object(agent, "_project_root_dir", return_value=project_root):
        with unittest.mock.patch.object(agent, "_root_work_dir", return_value=work_dir):
            expanded = agent._expand_bash_aliases("cat root/root/file.txt")

            assert expanded == "cat root/root/file.txt"

            expanded_multi = agent._expand_bash_aliases("ls root/a root/b")
            assert expanded_multi == "ls root/a root/b"
