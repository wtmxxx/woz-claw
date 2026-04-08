from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_memory_demo.memory_store import MemoryStore
from agent_memory_demo.service import ChatService


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


class CreateConversationRequest(BaseModel):
    user_id: str


app = FastAPI(title="Agent Memory Demo")

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
