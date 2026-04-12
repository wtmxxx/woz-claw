from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import yaml

from wozclaw.config import load_path_config
from wozclaw.memory_store import MemoryStore
from wozclaw.service import ChatService


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


class CreateConversationRequest(BaseModel):
    user_id: str


class RenameConversationRequest(BaseModel):
    user_id: str
    title: str


class SkillToggle(BaseModel):
    name: str
    enabled: bool


class OperationPolicy(BaseModel):
    read: str = "allow"
    write: str = "ask_human"
    delete: str = "ask_human"
    exec: str = "ask_human"


class CommandPolicy(BaseModel):
    enabled: bool = True
    default_action: str = "ask_human"
    operations: OperationPolicy = OperationPolicy()
    allowed_paths: list[str] = []
    command_allowlist: list[str] = []
    command_blocklist: list[str] = []


class SaveSettingsRequest(BaseModel):
    user_id: str
    long_term_memory: str
    skills: list[SkillToggle] = []
    command_policy: CommandPolicy | None = None


class ApprovalDecisionRequest(BaseModel):
    user_id: str
    session_id: str
    request_id: str
    approved: bool


class ChoiceDecisionRequest(BaseModel):
    user_id: str
    session_id: str
    request_id: str
    selected_option: str = ""
    custom_input: str = ""


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
        "approval_request": result.approval_request,
        "choice_request": result.choice_request,
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


@app.delete("/api/conversations/{session_id}")
def delete_conversation(user_id: str, session_id: str) -> dict[str, bool]:
    try:
        deleted = chat_service.delete_conversation(user_id, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "deleted": deleted}


@app.patch("/api/conversations/{session_id}/title")
def rename_conversation(session_id: str, req: RenameConversationRequest) -> dict[str, Any]:
    try:
        title = chat_service.rename_conversation(
            req.user_id,
            session_id,
            req.title,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "title": title}


@app.get("/api/conversations/{session_id}/messages")
def get_conversation_messages(user_id: str, session_id: str) -> dict[str, list[dict[str, Any]]]:
    try:
        rows = chat_service.get_session_messages(user_id, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"items": rows}


@app.get("/api/sessions/{session_id}/pending-state")
def get_pending_state(user_id: str, session_id: str) -> dict[str, Any]:
    """Get pending approvals and choices for a session."""
    user_text = user_id.strip()
    session_text = session_id.strip()
    if not user_text or not session_text:
        raise HTTPException(
            status_code=400,
            detail="user_id and session_id are required",
        )

    try:
        pending_approvals = memory_store.get_pending_approvals(
            user_text, session_text)
        pending_choices = memory_store.get_pending_choices(
            user_text, session_text)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=500, detail=f"server error: {exc}") from exc

    return {
        "pending_approvals": pending_approvals,
        "pending_choices": pending_choices,
    }


def _skills_root_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    path_cfg = load_path_config(project_root / "config" / "path.yaml")
    raw_dir = path_cfg.wozclaw_dir.strip() or ".wozclaw"
    base_dir = Path(raw_dir)
    if not base_dir.is_absolute():
        base_dir = project_root / base_dir
    return base_dir / "skills"


def _user_skills_yaml_path(user_id: str) -> Path:
    return _skills_root_dir() / user_id / "skills.yaml"


def _normalize_skill_zip_name(zip_filename: str) -> str:
    stem = Path(zip_filename).stem.strip()
    normalized = re.sub(r"(?i)[-_]?v?\d+(?:\.\d+)+$", "", stem).strip("-_. ")
    return normalized or stem


def _read_skill_name(skill_md_path: Path) -> str | None:
    try:
        text = skill_md_path.read_text(encoding="utf-8")
    except Exception:
        return None

    if not text.startswith("---"):
        return None

    parts = text.split("---", 2)
    if len(parts) < 3:
        return None

    try:
        loaded = yaml.safe_load(parts[1])
    except Exception:
        return None

    if not isinstance(loaded, dict):
        return None

    name = str(loaded.get("name", "")).strip()
    return name or None


def _sync_user_skill_toggle(user_id: str, skill_name: str) -> None:
    path = _user_skills_yaml_path(user_id)
    existing = _load_user_skill_toggles(user_id)
    merged: list[dict[str, Any]] = []
    seen = False

    for item in existing:
        if str(item.get("name", "")).strip() == skill_name:
            merged.append({"name": skill_name, "enabled": True})
            seen = True
        else:
            merged.append(item)

    if not seen:
        merged.append({"name": skill_name, "enabled": True})

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump({"skills": merged},
                       allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


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
        "command_policy": memory_store.get_command_policy(user_text),
    }


@app.put("/api/settings")
def save_settings(req: SaveSettingsRequest) -> dict[str, bool]:
    user_text = req.user_id.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="user_id is required")

    memory_store.set_long_term_memory(user_text, req.long_term_memory)
    _save_user_skill_toggles(user_text, req.skills)
    if req.command_policy is not None:
        memory_store.set_command_policy(
            user_text, req.command_policy.model_dump())
    return {"ok": True}


@app.post("/api/approvals/decide")
def decide_approval(req: ApprovalDecisionRequest) -> dict[str, Any]:
    user_text = req.user_id.strip()
    session_text = req.session_id.strip()
    request_text = req.request_id.strip()
    if not user_text or not session_text or not request_text:
        raise HTTPException(
            status_code=400,
            detail="user_id, session_id and request_id are required",
        )

    item = memory_store.pop_pending_approval(
        user_text, session_text, request_text)
    if item is None:
        raise HTTPException(
            status_code=404, detail="approval request not found")

    command = str(item.get("command", "")).strip()
    if not command:
        raise HTTPException(status_code=400, detail="invalid approval payload")

    if req.approved:
        from wozclaw.agent import ReActMemoryAgent

        runtime_agent = ReActMemoryAgent(
            memory_store=memory_store,
            user_id=user_text,
            session_id=session_text,
        )
        output = runtime_agent.run_bash_command_after_approval(command)
    else:
        output = "rejected by human"
    result = chat_service.resume_after_approval(
        user_id=user_text,
        session_id=session_text,
        request_id=request_text,
        command=command,
        output=output,
        approved=bool(req.approved),
    )

    return {
        "ok": True,
        "reply": result.reply,
        "memory_hits": result.memory_hits,
        "title": result.title,
        "tool_calls": result.tool_calls,
        "loaded_skills": result.loaded_skills,
        "activity_traces": result.activity_traces,
        "approval_request": result.approval_request,
        "choice_request": result.choice_request,
    }


@app.post("/api/choices/submit")
def submit_choice(req: ChoiceDecisionRequest) -> dict[str, Any]:
    user_text = req.user_id.strip()
    session_text = req.session_id.strip()
    request_text = req.request_id.strip()
    if not user_text or not session_text or not request_text:
        raise HTTPException(
            status_code=400,
            detail="user_id, session_id and request_id are required",
        )

    item = memory_store.pop_pending_choice(
        user_text, session_text, request_text)
    if item is None:
        raise HTTPException(status_code=404, detail="choice request not found")

    question = str(item.get("question", "")).strip()
    selected = req.selected_option.strip()
    custom = req.custom_input.strip()
    final_choice = custom or selected
    if not final_choice:
        raise HTTPException(
            status_code=400, detail="choice content is required")

    choice_message = f"针对问题【{question}】我的选择是：{final_choice}" if question else f"我的选择是：{final_choice}"
    result = chat_service.chat(
        user_text,
        session_text,
        choice_message,
        llm_user_message=choice_message,
        use_latest_session_memory=True,
    )

    return {
        "ok": True,
        "reply": result.reply,
        "memory_hits": result.memory_hits,
        "title": result.title,
        "tool_calls": result.tool_calls,
        "loaded_skills": result.loaded_skills,
        "activity_traces": result.activity_traces,
        "approval_request": result.approval_request,
        "choice_request": result.choice_request,
    }


@app.post("/api/skills/upload")
async def upload_skill_zip(
    user_id: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    user_text = user_id.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="user_id is required")

    filename = Path(file.filename or "").name.strip()
    if not filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400, detail="only zip files are supported")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty archive")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        archive_path = temp_root / filename
        extract_root = temp_root / "extracted"
        extract_root.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(raw)

        with ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_name = str(member.filename).strip()
                if not member_name:
                    raise HTTPException(
                        status_code=400,
                        detail="invalid archive member",
                    )
                member_path = Path(member_name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise HTTPException(
                        status_code=400,
                        detail="unsafe archive path",
                    )
            archive.extractall(extract_root)

        skill_md_candidates = sorted(
            extract_root.rglob("SKILL.md"),
            key=lambda item: len(item.relative_to(extract_root).parts),
        )
        if not skill_md_candidates:
            raise HTTPException(
                status_code=400, detail="SKILL.md not found in archive")

        skill_md_path = skill_md_candidates[0]
        skill_dir = skill_md_path.parent
        skill_name = _read_skill_name(
            skill_md_path) or _normalize_skill_zip_name(filename)
        if not skill_name:
            raise HTTPException(
                status_code=400, detail="skill name is required")

        skills_root = _skills_root_dir()
        target_dir = skills_root / user_text / skill_name
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(skill_dir), str(target_dir))

    _sync_user_skill_toggle(user_text, skill_name)
    return {"ok": True, "name": skill_name, "dir": str(target_dir)}
