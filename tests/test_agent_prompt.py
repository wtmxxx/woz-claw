from agent_memory_demo.agent import ReActMemoryAgent
from agent_memory_demo.memory_store import MemoryStore
from pathlib import Path
from types import SimpleNamespace
import subprocess
import yaml


def test_memory_prompt_requires_full_memory_rewrite(tmp_path) -> None:
    agent = ReActMemoryAgent(memory_store=MemoryStore(
        root_dir=tmp_path), user_id="u1", session_id="s1")

    prompt = agent.build_system_prompt("[LONG_TERM]\n用户喜欢游泳")

    assert "完整" in prompt
    assert "不是增量" in prompt
    assert "remember_note" in prompt
    assert "旧信息" in prompt
    assert "搜索" in prompt
    assert "message_id" in prompt
    assert "get_session_window" in prompt
    assert "get_daily_window" in prompt
    assert "优先调用工具" in prompt
    assert "能调用工具就调用工具" in prompt
    assert "只有在工具确实无法解决问题时" in prompt
    assert "当前用户ID" in prompt
    assert "u1" in prompt


def test_agent_module_no_langchain_or_langgraph_imports() -> None:
    source = Path("src/agent_memory_demo/agent.py").read_text(encoding="utf-8")

    assert "from langchain_core" not in source
    assert "from langgraph" not in source
    assert "from langchain_openai" not in source


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


def test_skill_shell_command_rejects_unsafe_pattern(tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(memory_store=store, user_id="u6", session_id="s6")

    ok, reason = agent._validate_skill_shell_command("Get-ChildItem ..")

    assert ok is False
    assert "unsafe" in reason


def test_skill_shell_command_requires_utf8_encoding_for_content_commands(tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(memory_store=store, user_id="u6", session_id="s6")

    ok_get, reason_get = agent._validate_skill_shell_command(
        "Get-Content demo-user/demo-user-style/SKILL.md"
    )
    ok_set, reason_set = agent._validate_skill_shell_command(
        "Set-Content demo-user/demo-user-style/SKILL.md -Value 'x'"
    )

    assert ok_get is False
    assert "encoding" in reason_get.lower()
    assert ok_set is False
    assert "encoding" in reason_set.lower()

    ok_get_utf8, _ = agent._validate_skill_shell_command(
        "Get-Content demo-user/demo-user-style/SKILL.md -Encoding UTF8"
    )
    ok_set_utf8, _ = agent._validate_skill_shell_command(
        "Set-Content demo-user/demo-user-style/SKILL.md -Value 'x' -Encoding UTF8"
    )

    assert ok_get_utf8 is True
    assert ok_set_utf8 is True


def test_skill_shell_command_runs_in_skills_root(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(memory_store=store, user_id="u7", session_id="s7")

    skills_root = tmp_path / "skills-root"
    skills_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(agent, "_skills_root_dir", lambda: skills_root)

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

    result = agent._run_skill_shell_command("Get-ChildItem global")

    assert result == "ok"
    assert captured["cwd"] == skills_root
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["shell"] is False
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"


def test_skill_shell_command_forces_utf8_output_on_windows(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(memory_store=store, user_id="u8", session_id="s8")

    skills_root = tmp_path / "skills-root"
    skills_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(agent, "_skills_root_dir", lambda: skills_root)

    captured: dict[str, object] = {}

    def fake_run(cmd, cwd, capture_output, text, timeout, shell, encoding, errors):  # noqa: ANN001
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="中文测试", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = agent._run_skill_shell_command("echo 中文测试")

    assert result == "中文测试"
    assert isinstance(captured["cmd"], list)
    assert captured["cmd"][2] == "-Command"
    assert "OutputEncoding = [System.Text.Encoding]::UTF8" in captured["cmd"][3]
