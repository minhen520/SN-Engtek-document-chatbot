"""FastAPI server for the OpenAI-only RAG app.

Run with:  uvicorn server:app --reload
Then open: http://localhost:8000
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Must run before rag reads env vars. override=True so the project's .env
# wins over any stale OPENAI_API_KEY in the Windows environment.
load_dotenv(override=True)

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import APIStatusError, AuthenticationError, OpenAIError
from pydantic import BaseModel, Field

import rag

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB per file

app = FastAPI(title="OpenAI RAG", version="1.0.0")
STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = Path(__file__).parent / "data"
CHAT_HISTORY_FILE = DATA_DIR / "chat_history.json"
COMPANY_LOGO_DIR = Path(__file__).parent / "company_logo"


class _SessionState:
    """Short-lived conversation state shared by all providers."""

    __slots__ = ("history", "openai_response_id", "updated_at")

    def __init__(self) -> None:
        self.history: list[dict[str, str]] = []
        self.openai_response_id: str | None = None
        self.updated_at = time.time()


_sessions_lock = threading.Lock()
_sessions: dict[str, _SessionState] = {}
_SESSION_TTL_SECONDS = 60 * 60


def _clear_chat_history_file() -> None:
    try:
        CHAT_HISTORY_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _write_chat_history_file(session_id: str, history: list[dict[str, str]]) -> None:
    """Write a readable runtime snapshot; it is cleared on restart/reset."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        CHAT_HISTORY_FILE.write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "messages": history,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass


def _get_session(session_id: str) -> _SessionState:
    now = time.time()
    with _sessions_lock:
        expired = [sid for sid, state in _sessions.items() if now - state.updated_at > _SESSION_TTL_SECONDS]
        for sid in expired:
            _sessions.pop(sid, None)
        session = _sessions.setdefault(session_id, _SessionState())
        session.updated_at = now
        return session


# Disk history is a runtime aid, never a cross-restart data store.
_clear_chat_history_file()


@app.exception_handler(OpenAIError)
def openai_error(request: Request, exc: OpenAIError) -> JSONResponse:
    if isinstance(exc, AuthenticationError):
        detail = (
            "OpenAI rejected the API key (invalid or expired). "
            "Update OPENAI_API_KEY in your .env file."
        )
        status = 401
    elif isinstance(exc, APIStatusError):
        detail = f"OpenAI API error: {exc.message}"
        status = exc.status_code if exc.status_code >= 400 else 502
    else:
        detail = f"OpenAI API error: {exc}"
        status = 502
    return JSONResponse(status_code=status, content={"detail": detail})


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    previous_response_id: str | None = None
    session_id: str | None = None
    provider: Literal["openai", "lmstudio"] = "openai"


@app.get("/api/health")
def health(provider: Literal["openai", "lmstudio"] = "openai") -> dict:
    return rag.health(provider)


@app.post("/api/upload")
async def upload(
    files: list[UploadFile] = File(...),
    provider: Literal["openai", "lmstudio"] = "openai",
) -> dict:
    """Upload one or more documents; blocks until each is indexed."""
    results, errors = [], []
    for f in files:
        try:
            content = await f.read()
            if not content:
                raise ValueError("File is empty.")
            if len(content) > MAX_UPLOAD_BYTES:
                raise ValueError("File exceeds the 50 MB limit.")
            results.append(rag.upload_document(f.filename or "upload", content))
        except Exception as exc:  # report per-file failures, keep going
            errors.append({"filename": f.filename, "error": str(exc)})
    if not results and errors:
        raise HTTPException(status_code=400, detail=errors)
    return {"uploaded": results, "errors": errors}


@app.get("/api/documents")
def documents(provider: Literal["openai", "lmstudio"] = "openai") -> dict:
    try:
        return {"documents": rag.list_documents()}
    except RuntimeError as exc:  # e.g. missing API key
        raise HTTPException(status_code=503, detail=str(exc))


@app.delete("/api/documents/{file_id}")
def delete_document(
    file_id: str,
    provider: Literal["openai", "lmstudio"] = "openai",
) -> dict:
    rag.delete_document(file_id)
    return {"deleted": file_id}


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    provider = req.provider
    try:
        question = req.question.strip()
        session_id = (req.session_id or "").strip()
        if not session_id:
            return rag.ask(question, req.previous_response_id, provider=provider)

        session = _get_session(session_id)
        response_id = req.previous_response_id
        if provider == "openai" and not response_id:
            response_id = session.openai_response_id

        result = rag.ask(
            question,
            previous_response_id=response_id,
            provider=provider,
            history=session.history,
        )

        session.history.append({"role": "user", "content": question})
        answer = (result.get("answer") or "").strip()
        if answer:
            session.history.append({"role": "assistant", "content": answer})
        session.updated_at = time.time()
        _write_chat_history_file(session_id, session.history)

        if provider == "openai":
            returned_id = result.get("response_id")
            if isinstance(returned_id, str) and returned_id:
                session.openai_response_id = returned_id
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/session/reset")
def reset_session(payload: dict) -> dict:
    session_id = payload.get("session_id") if isinstance(payload, dict) else None
    if not isinstance(session_id, str) or not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    with _sessions_lock:
        _sessions.pop(session_id.strip(), None)
    _clear_chat_history_file()
    return {"reset": True}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/assets", StaticFiles(directory=COMPANY_LOGO_DIR, check_dir=False), name="assets")
