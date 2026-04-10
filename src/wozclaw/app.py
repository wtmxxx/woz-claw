from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import yaml

from wozclaw.memory_store import MemoryStore
from wozclaw.service import ChatService


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


class CreateConversationRequest(BaseModel):
    user_id: str


class SkillToggle(BaseModel):
    name: str
    enabled: bool


class SaveSettingsRequest(BaseModel):
    user_id: str
    long_term_memory: str
    skills: list[SkillToggle] = []


app = FastAPI(title="WozClaw")

memory_store = MemoryStore(root_dir=Path("memory"))
chat_service = ChatService(memory_store=memory_store)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.post("/api/chat")
def chat_api(req: ChatRequest) -> dict[str, Any]:
    try:
        result = chat_service.chat(req.user_id, req.session_id, req.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500, detail=f"server error: {exc}") from exc

    return {
        "reply": result.reply,
        "memory_hits": result.memory_hits,
        "title": result.title,
        "tool_calls": result.tool_calls,
        "loaded_skills": result.loaded_skills,
        "activity_traces": result.activity_traces,
    }


@app.post("/api/conversations")
def create_conversation(req: CreateConversationRequest) -> dict[str, str]:
    try:
        session_id = chat_service.create_conversation(req.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"session_id": session_id}


@app.get("/api/conversations")
def list_conversations(user_id: str) -> dict[str, list[dict[str, str]]]:
    try:
        rows = chat_service.list_conversations(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"items": rows}


@app.get("/api/conversations/{session_id}/messages")
def get_conversation_messages(user_id: str, session_id: str) -> dict[str, list[dict[str, Any]]]:
    try:
        rows = chat_service.get_session_messages(user_id, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"items": rows}


def _skills_root_dir() -> Path:
    return Path(__file__).resolve().parents[2] / ".sandbox" / "skills"


def _user_skills_yaml_path(user_id: str) -> Path:
    return _skills_root_dir() / user_id / "skills.yaml"


def _load_user_skill_toggles(user_id: str) -> list[dict[str, Any]]:
    path = _user_skills_yaml_path(user_id)
    if not path.exists():
        return []

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(loaded, dict):
        return []
    raw_skills = loaded.get("skills", [])
    if not isinstance(raw_skills, list):
        return []

    result: list[dict[str, Any]] = []
    for item in raw_skills:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        result.append(
            {"name": name, "enabled": bool(item.get("enabled", True))})
    return result


def _save_user_skill_toggles(user_id: str, skills: list[SkillToggle]) -> None:
    path = _user_skills_yaml_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "skills": [
            {"name": item.name.strip(), "enabled": bool(item.enabled)}
            for item in skills
            if item.name.strip()
        ]
    }
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


@app.get("/api/settings")
def get_settings(user_id: str) -> dict[str, Any]:
    user_text = user_id.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="user_id is required")

    return {
        "long_term_memory": memory_store.get_long_term_memory(user_text),
        "skills": _load_user_skill_toggles(user_text),
    }


@app.put("/api/settings")
def save_settings(req: SaveSettingsRequest) -> dict[str, bool]:
    user_text = req.user_id.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="user_id is required")

    memory_store.set_long_term_memory(user_text, req.long_term_memory)
    _save_user_skill_toggles(user_text, req.skills)
    return {"ok": True}
