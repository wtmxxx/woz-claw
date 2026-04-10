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


@dataclass
class AgentRuntimeConfig:
    skill_dirs: list[str]
    memory_group_active: bool
    skill_instruction: str
    skill_template: str


@dataclass
class SandboxConfig:
    writable_dir: str


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


def load_agent_runtime_config(config_path: Path | str = "config/agent.yaml") -> AgentRuntimeConfig:
    path = Path(config_path)
    data: dict[str, Any] = {}

    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded

    agent_data = data.get("agent", {}) if isinstance(
        data.get("agent", {}), dict) else {}

    raw_skill_dirs = agent_data.get("skill_dirs", [])
    skill_dirs: list[str] = []
    if isinstance(raw_skill_dirs, list):
        for item in raw_skill_dirs:
            if isinstance(item, str) and item.strip():
                skill_dirs.append(item.strip())

    return AgentRuntimeConfig(
        skill_dirs=skill_dirs,
        memory_group_active=bool(agent_data.get("memory_group_active", True)),
        skill_instruction=str(
            agent_data.get(
                "skill_instruction",
                "你可以使用 Agent Skills。每个技能目录中都有 SKILL.md，先遵循技能指引再调用工具。",
            )
        ),
        skill_template=str(
            agent_data.get(
                "skill_template",
                "- {name}({dir}): {description}",
            )
        ),
    )


def load_sandbox_config(config_path: Path | str = "config/sandbox.yaml") -> SandboxConfig:
    path = Path(config_path)
    data: dict[str, Any] = {}

    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded

    sandbox_data = data.get("sandbox", {}) if isinstance(
        data.get("sandbox", {}), dict) else {}

    return SandboxConfig(
        writable_dir=str(sandbox_data.get("writable_dir", "")).strip(),
    )
