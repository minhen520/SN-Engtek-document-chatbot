"""FastAPI server for the OpenAI-only RAG app.

Run with:  uvicorn server:app --reload
Then open: http://localhost:8000
"""

from __future__ import annotations

import os
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
        return rag.ask(req.question.strip(), req.previous_response_id, provider=provider)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
