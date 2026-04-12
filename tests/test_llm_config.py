from pathlib import Path

from wozclaw.config import (
    load_agent_runtime_config,
    load_llm_config,
    load_path_config,
)


def test_load_llm_config_from_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "llm.yaml"
    config_file.write_text(
        """
llm:
    api_key: test-key
    model: test-model
    base_url: https://example.com/v1
    compact_model: test-compact-model
""".strip(),
        encoding="utf-8",
    )

    config = load_llm_config(config_file)

    assert config.api_key == "test-key"
    assert config.model == "test-model"
    assert config.base_url == "https://example.com/v1"
    assert config.compact_model == "test-compact-model"


def test_load_llm_config_falls_back_to_defaults_when_file_missing(tmp_path: Path) -> None:
    config = load_llm_config(tmp_path / "missing.yaml")

    assert config.model == "gpt-4o-mini"
    assert config.api_key == ""
    assert config.base_url == ""
    assert config.compact_model == ""


def test_load_agent_runtime_config_from_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "agent.yaml"
    config_file.write_text(
        """
agent:
    skill_dirs:
        - src/wozclaw/skills/memory-tools
    memory_group_active: false
    skill_instruction: custom instruction
    skill_template: "- {name}: {description}"
""".strip(),
        encoding="utf-8",
    )

    config = load_agent_runtime_config(config_file)

    assert config.skill_dirs == ["src/wozclaw/skills/memory-tools"]
    assert config.memory_group_active is False
    assert config.skill_instruction == "custom instruction"
    assert config.skill_template == "- {name}: {description}"


def test_load_agent_runtime_config_defaults_when_file_missing(tmp_path: Path) -> None:
    config = load_agent_runtime_config(tmp_path / "missing-agent.yaml")

    assert config.skill_dirs == []
    assert config.memory_group_active is True
    assert "Agent Skills" in config.skill_instruction
    assert "{name}" in config.skill_template


def test_load_path_config_from_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "path.yaml"
    config_file.write_text(
        """
path:
  workdir: workdir
  wozclaw_dir: .wozclaw-custom
""".strip(),
        encoding="utf-8",
    )

    config = load_path_config(config_file)

    assert config.workdir == str((tmp_path / "workdir").resolve())
    assert config.wozclaw_dir == str((tmp_path / ".wozclaw-custom").resolve())


def test_load_path_config_defaults_when_missing(tmp_path: Path) -> None:
    config = load_path_config(tmp_path / "missing-path.yaml")

    assert config.workdir == ""
    assert config.wozclaw_dir == str((tmp_path / ".wozclaw").resolve())


def test_load_path_config_resolves_relative_paths_against_project_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project-root"
    config_dir = project_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "path.yaml"
    config_file.write_text(
        """
path:
  workdir: workdir
  wozclaw_dir: .wozclaw
""".strip(),
        encoding="utf-8",
    )

    config = load_path_config(config_file)

    assert config.workdir == str((project_root / "workdir").resolve())
    assert config.wozclaw_dir == str((project_root / ".wozclaw").resolve())
