from wozclaw.agent import ReActMemoryAgent
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
    assert "bash_command" in prompt
    assert "root/" in prompt
    assert "cat root/README.md" in prompt
    assert "cat .sandbox/skills/demo-user/SKILL.md" in prompt
    assert "root/是工作目录" not in prompt  # Changed phrasing
    assert ".sandbox/是Agent配置" not in prompt  # Changed phrasing
    assert "root/ 表示工作目录" in prompt
    assert ".sandbox/ 表示项目根目录" in prompt
    assert "输出最多保留100000字符" in prompt
    assert "尽量不要一次读取大段内容" in prompt


def test_agent_module_no_langchain_or_langgraph_imports() -> None:
    source = Path("src/wozclaw/agent.py").read_text(encoding="utf-8")

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


def test_bash_command_only_disallows_dotdot(tmp_path) -> None:
    """Test that only .. traversal is forbidden now."""
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(memory_store=store, user_id="u6", session_id="s6")

    project_root = tmp_path / "project-root"
    (project_root / ".sandbox").mkdir(parents=True, exist_ok=True)
    monkeypatch = __import__("unittest.mock", fromlist=[
                             "MagicMock"]).MagicMock()

    # Paths within workspace should pass now
    ok, reason = agent._validate_bash_command("ls src")
    assert ok is True

    # Only .. should fail
    ok, reason = agent._validate_bash_command("ls ../outside")
    assert ok is False
    assert "unsafe" in reason

    # Literal text with .. should not be treated as traversal
    ok, reason = agent._validate_bash_command('echo "a..b"')
    assert ok is True
    assert reason == ""

    # Quoted traversal is treated as literal text
    ok, reason = agent._validate_bash_command('ls "../outside"')
    assert ok is True
    assert reason == ""


def test_bash_path_validation_only_blocks_actual_traversal(tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u6b", session_id="s6b")

    ok, reason = agent._validate_bash_path("../outside")
    assert ok is False
    assert "unsafe" in reason

    ok, reason = agent._validate_bash_path("a..b")
    assert ok is True
    assert reason == ""


def test_bash_command_runs_in_default_sandbox_with_bash(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(memory_store=store, user_id="u7", session_id="s7")

    project_root = tmp_path / "project-root"
    work_dir = project_root / "custom-sandbox"
    default_sandbox = project_root / ".sandbox"
    default_sandbox.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(agent, "_project_root_dir", lambda: project_root)

    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "sandbox.yaml").write_text(
        "sandbox:\n  writable_dir: custom-sandbox\n",
        encoding="utf-8",
    )

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

    result = agent._run_bash_command("ls .sandbox")

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

    project_root = tmp_path / "project-root"
    default_sandbox = project_root / ".sandbox"
    configured_sandbox = project_root / "custom-sandbox"
    default_sandbox.mkdir(parents=True, exist_ok=True)
    configured_sandbox.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(agent, "_project_root_dir", lambda: project_root)

    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "sandbox.yaml").write_text(
        "sandbox:\n  writable_dir: custom-sandbox\n",
        encoding="utf-8",
    )

    expanded = agent._expand_bash_aliases(
        "ls root/src && ls .sandbox"
    )

    assert (configured_sandbox / "src").as_posix() in expanded
    assert default_sandbox.as_posix() in expanded
    assert expanded.count(default_sandbox.as_posix()) == 1


def test_bash_command_expands_dot_sandbox_without_double_replacement(monkeypatch, tmp_path) -> None:
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u9b", session_id="s9b")

    project_root = tmp_path / "project-root"
    default_sandbox = project_root / ".sandbox"
    default_sandbox.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(agent, "_project_root_dir", lambda: project_root)

    expanded = agent._expand_bash_aliases("ls -la .sandbox/")

    default_sandbox_posix = default_sandbox.resolve().as_posix()
    assert default_sandbox_posix in expanded
    assert expanded.count(default_sandbox_posix) == 1


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
    """Test that bash output paths are converted from work_dir to root/ format."""
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

    # Should convert work_dir prefix to root/
    assert "root/src" in result
    assert work_dir_posix not in result


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


def test_configured_sandbox_handles_unix_style_paths(monkeypatch, tmp_path) -> None:
    """Test that writable_dir with Unix-style path (e.g., /workdir) is resolved correctly."""
    store = MemoryStore(root_dir=tmp_path)
    agent = ReActMemoryAgent(
        memory_store=store, user_id="u13", session_id="s13")

    project_root = tmp_path / "project-root"
    project_root.mkdir(parents=True, exist_ok=True)
    workdir = project_root / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "test.txt").write_text("content", encoding="utf-8")

    monkeypatch.setattr(agent, "_project_root_dir", lambda: project_root)

    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "sandbox.yaml").write_text(
        "sandbox:\n  writable_dir: /workdir\n",
        encoding="utf-8",
    )

    # Force reload of config by calling the method directly
    configured = agent._configured_sandbox_dir()
    assert configured == workdir

    # Verify root_work_dir also returns the correct path
    root_work = agent._root_work_dir()
    assert root_work == workdir


def test_bash_aliases_preserve_pipes_and_redirects(tmp_path) -> None:
    """Test that bash operators (pipes, redirects, etc.) are preserved during path expansion."""
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

            # Verify paths were still replaced
            work_dir_posix = work_dir.resolve().as_posix()
            assert work_dir_posix in expanded_pipe


def test_nested_root_prefix_not_double_replaced(tmp_path) -> None:
    """Test that root/root/ is replaced only once to workspace/root/, not workspace/workspace/.

    This addresses the issue: if workdir contains a 'root/' subdirectory,
    then a command like 'cat root/root/file.txt' should become 'cat <workdir-path>/root/file.txt',
    not 'cat <workdir-path>/<workdir-path>/file.txt'.
    """
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
            work_dir_posix = work_dir.resolve().as_posix()

            # Test: root/root/file should become workdir/root/file, NOT workdir/workdir/file
            expanded = agent._expand_bash_aliases("cat root/root/file.txt")

            # Should contain workdir path once
            assert expanded.count(work_dir_posix) == 1, \
                f"Path should appear exactly once, but got: {expanded}"

            # Should NOT contain double workdir
            double_path = work_dir_posix + "/" + work_dir_posix
            assert double_path not in expanded, \
                f"Path was double-replaced: {expanded}"

            # Should have /root/ in the middle
            assert "/root/" in expanded or "\\root\\" in expanded, \
                f"Nested root/ should remain: {expanded}"

            # Test with multiple root/ references: root/a root/b should both be replaced
            expanded_multi = agent._expand_bash_aliases("ls root/a root/b")
            assert expanded_multi.count(work_dir_posix) == 2, \
                f"Both root/ references should be replaced: {expanded_multi}"
