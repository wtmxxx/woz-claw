from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LLMConfig:
    api_key: str
    model: str
    base_url: str


def load_llm_config(config_path: Path | str = "config/llm.yaml") -> LLMConfig:
    path = Path(config_path)
    data: dict[str, Any] = {}

    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded

    llm_data = data.get("llm", {}) if isinstance(
        data.get("llm", {}), dict) else {}

    return LLMConfig(
        api_key=str(llm_data.get("api_key", "")),
        model=str(llm_data.get("model", "gpt-4o-mini")),
        base_url=str(llm_data.get("base_url", "")),
    )
