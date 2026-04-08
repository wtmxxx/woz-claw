from pathlib import Path

from agent_memory_demo.config import load_llm_config


def test_load_llm_config_from_yaml(tmp_path: Path) -> None:
    config_file = tmp_path / "llm.yaml"
    config_file.write_text(
        """
llm:
  api_key: test-key
  model: test-model
  base_url: https://example.com/v1
""".strip(),
        encoding="utf-8",
    )

    config = load_llm_config(config_file)

    assert config.api_key == "test-key"
    assert config.model == "test-model"
    assert config.base_url == "https://example.com/v1"


def test_load_llm_config_falls_back_to_defaults_when_file_missing(tmp_path: Path) -> None:
    config = load_llm_config(tmp_path / "missing.yaml")

    assert config.model == "gpt-4o-mini"
    assert config.api_key == ""
    assert config.base_url == ""
