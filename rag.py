"""OpenAI-only RAG engine.

Everything — chunking, embedding, vector storage, retrieval, and answer
generation — is handled by the OpenAI API:

  * Vector Stores API  -> document ingestion (chunk + embed + index) and
                          hybrid semantic/keyword retrieval with reranking.
  * Responses API      -> grounded answer generation via the `file_search`
                          tool, returning citations anchored to the answer.

No local embedding math, no third-party vector database.
"""

from __future__ import annotations

import io
import json
import os
import threading
import re
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI, NotFoundError

try:
    from tavily import TavilyClient  # type: ignore
except Exception:  # pragma: no cover
    TavilyClient = None  # type: ignore

DATA_DIR = Path(__file__).parent / "data"
STATE_FILE = DATA_DIR / "state.json"

CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "gpt-4o-mini")
VECTOR_STORE_NAME = os.getenv("RAG_VECTOR_STORE_NAME", "rag-app-store")
MAX_RESULTS = int(os.getenv("RAG_MAX_RESULTS", "8"))
MIN_DOC_RELEVANCE = float(os.getenv("RAG_MIN_DOC_RELEVANCE", "0.2"))

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL")

WEB_SEARCH_ENABLED = os.getenv("RAG_WEB_SEARCH_ENABLED", "0") == "1"
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_MAX_RESULTS = int(os.getenv("RAG_TAVILY_MAX_RESULTS", "5"))

DEBUG = os.getenv("RAG_DEBUG", "0") == "1"
_log = logging.getLogger("rag")

# File types accepted by OpenAI file_search.
SUPPORTED_EXTENSIONS = {
    ".c", ".cpp", ".cs", ".css", ".doc", ".docx", ".go", ".html", ".java",
    ".js", ".json", ".md", ".pdf", ".php", ".pptx", ".py", ".rb", ".sh",
    ".tex", ".ts", ".txt",
}

SYSTEM_PROMPT = """\
You are an assistant in a document chatbot.

You may be given:
- DOCUMENT PASSAGES (internal uploaded documents)
- WEB SOURCES (from web search)

Rules:
- If DOCUMENT PASSAGES are provided, answer using ONLY those passages.
- If WEB SOURCES are provided, answer using ONLY those sources.
- If neither are provided, you may answer from your general training/knowledge.
- Be concise and direct. If unsure, say so plainly.

Safety & scope:
- For personal HR/account-specific requests or system-dependent internal operational/financial requests:
    - If DOCUMENT PASSAGES explicitly contain the requested information, you may answer and cite it.
    - Otherwise, do NOT guess. Politely refuse and direct the user to the appropriate team/system.

Output format:
- If the user message asks for JSON with fields {answer, confidence}, return JSON only.
    confidence must be a number from 0 to 1.
"""

_lock = threading.Lock()
_client: OpenAI | None = None
_lm_client: OpenAI | None = None

Provider = Literal["openai", "lmstudio"]


def client() -> OpenAI:
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = OpenAI()
    return _client


def lm_client() -> OpenAI:
    """OpenAI-compatible client for LM Studio."""
    global _lm_client
    if _lm_client is None:
        # LM Studio typically ignores the key but the OpenAI SDK requires one.
        _lm_client = OpenAI(base_url=LM_STUDIO_URL, api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"))
    return _lm_client


def _resolve_lm_studio_model() -> str:
    if LM_STUDIO_MODEL and LM_STUDIO_MODEL.strip():
        return LM_STUDIO_MODEL.strip()
    # Try to pick the first available model from the local server.
    models = lm_client().models.list()
    for m in getattr(models, "data", []) or []:
        mid = getattr(m, "id", None)
        if isinstance(mid, str) and mid.strip():
            return mid.strip()
    raise RuntimeError(
        "LM_STUDIO_MODEL is not set and no models were returned by LM Studio. "
        "In LM Studio, load a model and enable the OpenAI-compatible server."
    )


def health(provider: Provider = "openai") -> dict[str, Any]:
    if provider == "openai":
        return {
            "ok": True,
            "provider": "openai",
            "api_key_set": bool(os.getenv("OPENAI_API_KEY")),
            "model": CHAT_MODEL,
        }
    try:
        model = _resolve_lm_studio_model()
        return {
            "ok": True,
            "provider": "lmstudio",
            "api_key_set": False,
            "model": model,
            "base_url": LM_STUDIO_URL,
        }
    except Exception as exc:
        return {
            "ok": False,
            "provider": "lmstudio",
            "api_key_set": False,
            "detail": str(exc),
            "base_url": LM_STUDIO_URL,
        }


def _normalize(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())

def _parse_json_answer(text: str) -> tuple[str, float] | None:
    """Best-effort parse for {answer, confidence}."""
    if not text:
        return None
    t = text.strip()
    # Try strict JSON first.
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
            conf = obj.get("confidence")
            if not isinstance(conf, (int, float)):
                conf = 0.5
            return obj["answer"].strip(), float(conf)
    except Exception:
        pass

    # Fallback: extract a JSON object substring.
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict) and isinstance(obj.get("answer"), str):
            conf = obj.get("confidence")
            if not isinstance(conf, (int, float)):
                conf = 0.5
            return obj["answer"].strip(), float(conf)
    except Exception:
        return None
    return None


def _question_about_model(question: str) -> bool:
    q = _normalize(question)
    if not q:
        return False
    return bool(re.search(r"\b(model|llm|gpt)\b", q) and re.search(r"\b(what|which|ur|your)\b", q))


def _openai_response_text(resp: Any) -> str:
    parts: list[str] = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", []) or []:
            if getattr(part, "type", None) == "output_text":
                parts.append(getattr(part, "text", ""))
    return "\n".join(p.strip() for p in parts if p and p.strip()).strip()


def _llm_answer_json(
    provider: Provider,
    prompt: str,
    previous_response_id: str | None = None,
) -> tuple[str, float, str | None]:
    """Return (answer, confidence, response_id)."""
    if provider == "openai":
        resp = client().responses.create(
            model=CHAT_MODEL,
            instructions=SYSTEM_PROMPT,
            input=prompt,
            previous_response_id=previous_response_id,
        )
        raw = _openai_response_text(resp)
        parsed = _parse_json_answer(raw)
        if parsed is None:
            return raw or "(no answer)", 0.5, getattr(resp, "id", None)
        answer, conf = parsed
        conf = max(0.0, min(1.0, float(conf)))
        return answer or "(no answer)", conf, getattr(resp, "id", None)

    model = _resolve_lm_studio_model()
    completion = lm_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = (
        completion.choices[0].message.content
        if completion.choices and completion.choices[0].message
        else ""
    )
    parsed = _parse_json_answer(raw or "")
    if parsed is None:
        return (raw or "").strip() or "(no answer)", 0.5, None
    answer, conf = parsed
    conf = max(0.0, min(1.0, float(conf)))
    return answer or "(no answer)", conf, None


def _build_numbered_passages(sources: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Return (context_text, citations). Citations are unique per file_id."""
    citations: list[dict[str, Any]] = []
    cite_index: dict[str, int] = {}
    blocks: list[str] = []

    for s in sources:
        fid = s.get("file_id")
        if not isinstance(fid, str) or not fid:
            continue
        if fid not in cite_index:
            cite_index[fid] = len(cite_index) + 1
            citations.append(
                {
                    "n": cite_index[fid],
                    "file_id": fid,
                    "filename": s.get("filename"),
                }
            )
        n = cite_index[fid]
        name = s.get("filename") or fid
        text = (s.get("text") or s.get("snippet") or "").strip()
        blocks.append(f"[{n}] {name}\n{text}")

    return "DOCUMENT PASSAGES:\n\n" + "\n\n".join(blocks), citations


def _retrieve_doc_sources(question: str, previous_response_id: str | None = None) -> tuple[list[dict[str, Any]], str]:
    """Retrieve doc passages via OpenAI file_search_call results."""
    vs_id = get_vector_store_id()
    resp = client().responses.create(
        model=CHAT_MODEL,
        instructions=(
            "Use file_search to retrieve passages that answer the user's question. "
            "Then output the single token OK."
        ),
        input=question,
        previous_response_id=previous_response_id,
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vs_id],
                "max_num_results": MAX_RESULTS,
            }
        ],
        include=["file_search_call.results"],
    )

    sources: list[dict[str, Any]] = []
    for item in resp.output:
        if item.type != "file_search_call":
            continue
        for r in item.results or []:
            text = (r.text or "").strip()
            snippet = text
            if len(snippet) > 600:
                snippet = snippet[:600].rsplit(" ", 1)[0] + " …"
            sources.append(
                {
                    "file_id": r.file_id,
                    "filename": r.filename,
                    "score": round(r.score, 3) if r.score is not None else None,
                    "text": text,
                    "snippet": snippet,
                }
            )
    return sources, resp.id


def _sensitive_internal_refusal(question: str) -> str | None:
    """Return a refusal message if the question is personal/internal operational."""
    q = _normalize(question)
    if not q:
        return None

    # Personal HR / payroll / identity-specific
    hr_hits = [
        "annual leave", "leave balance", "remaining leave", "leave left", "vacation balance",
        "payslip", "pay slip", "salary", "payroll", "epf", "socso", "tax", "bonus",
        "claim status", "reimbursement status",
    ]

    # Finance/procurement/costs (usually system-specific)
    finance_hits = [
        "equipment cost", "how much does the equipment cost", "purchase cost", "unit cost",
        "budget", "invoice", "quotation", "quote", "procurement", "purchase order", "po",
        "expense", "billing", "price approval",
    ]

    # If user asks "my" / "i" plus an HR keyword, treat as personal.
    if any(k in q for k in hr_hits) and re.search(r"\b(my|mine|i|me)\b", q):
        return (
            "I can’t help with personal HR/account details like leave balance or payroll. "
            "Please contact your HR/People Ops team (or your HR portal/helpdesk) for this request."
        )

    # Even without "my", some questions are inherently system-specific.
    if any(k in q for k in ("how many annual leave", "annual leave left", "leave left", "leave balance")):
        return (
            "I can’t determine leave balances because that requires access to HR systems. "
            "Please check your HR portal or contact HR/People Ops."
        )

    if any(k in q for k in finance_hits):
        return (
            "I can’t provide internal cost/budget/procurement figures unless they are explicitly in your uploaded documents. "
            "Please contact Finance/Procurement (or the asset owner) for the official numbers."
        )

    return None


def _is_smalltalk(question: str) -> bool:
    q = _normalize(question)
    if not q:
        return False
    patterns = [
        r"^hi$|^hello$|^hey$|^good (morning|afternoon|evening)$",
        r"how are you\??$|how r u\??$",
        r"how was your day\??$|how's your day\??$",
        r"thank you\.?$|thanks\.?$",
    ]
    return any(re.search(p, q) for p in patterns)


def can_use_web_search(question: str) -> bool:
    if not WEB_SEARCH_ENABLED:
        return False
    if not TAVILY_API_KEY:
        return False
    if TavilyClient is None:
        return False
    if _sensitive_internal_refusal(question) is not None:
        return False
    return True


def _answer_says_not_found(answer: str) -> bool:
    a = _normalize(answer)
    if not a:
        return False
    phrases = [
        "documents you provided do not contain",
        "uploaded documents do not contain",
        "documents do not contain",
        "the documents don't contain",
        "the documents do not contain",
        "i don't have enough information in the documents",
        "i do not have enough information in the documents",
        "i cannot find this in the documents",
    ]
    return any(p in a for p in phrases)


def _has_document_match(sources: list[dict[str, Any]]) -> bool:
    """Return True if retrieval found a meaningfully relevant match."""
    if not sources:
        return False
    scores = [s.get("score") for s in sources if isinstance(s.get("score"), (int, float))]
    if not scores:
        # If we can't score, assume the presence of sources implies a match.
        return True
    return max(scores) >= MIN_DOC_RELEVANCE


def has_documents() -> bool:
    """Fast check whether the vector store contains any indexed files."""
    vs_id = get_vector_store_id()
    for _ in client().vector_stores.files.list(vector_store_id=vs_id, limit=1):
        return True
    return False


def _tavily_search(question: str) -> list[dict[str, Any]]:
    if not can_use_web_search(question):
        return []
    tv = TavilyClient(api_key=TAVILY_API_KEY)
    result = tv.search(
        query=question,
        max_results=max(1, min(TAVILY_MAX_RESULTS, 10)),
        include_answer=False,
        include_raw_content=False,
    )
    out: list[dict[str, Any]] = []
    for r in (result.get("results") or []):
        url = (r.get("url") or "").strip()
        title = (r.get("title") or url or "Source").strip()
        snippet = (r.get("content") or "").strip()
        if snippet and len(snippet) > 800:
            snippet = snippet[:800].rsplit(" ", 1)[0] + " …"
        out.append({"title": title, "url": url, "snippet": snippet})
    if DEBUG:
        _log.info("tavily: results=%s", len(out))
    return out


def _ask_with_web_sources(
    question: str,
    provider: Provider,
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    sources = _tavily_search(question)
    if not sources:
        return {
            "answer": "I can’t answer that from your uploaded documents, and web search is not enabled.",
            "citations": [],
            "sources": [],
            "response_id": previous_response_id,
            "confidence": 0.0,
        }

    numbered = []
    for i, s in enumerate(sources, start=1):
        numbered.append(f"[{i}] {s['title']} — {s['url']}\n{s['snippet']}")
    context = "WEB SOURCES:\n\n" + "\n\n".join(numbered)

    user = (
        "Use ONLY the provided WEB SOURCES. Cite every factual claim with [n] where n is the source number.\n"
        "Return JSON only as {\"answer\": string, \"confidence\": number} (0..1).\n\n"
        f"Question: {question}\n\n{context}"
    )

    answer, conf, resp_id = _llm_answer_json(provider, user, previous_response_id=previous_response_id)

    citations = [
        {"n": i, "file_id": s["url"], "filename": s["title"]}
        for i, s in enumerate(sources, start=1)
    ]
    ui_sources = [
        {"file_id": s["url"], "filename": s["title"], "score": None, "snippet": s["snippet"]}
        for s in sources
    ]
    return {
        "answer": (answer or "").strip() or "(no answer)",
        "citations": citations,
        "sources": ui_sources,
        "response_id": resp_id,
        "confidence": conf,
    }


# --------------------------------------------------------------------------
# Local state: just the vector store id, so restarts reuse the same store.
# --------------------------------------------------------------------------

def _load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def _save_state(state: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get_vector_store_id() -> str:
    """Return the app's vector store id, creating the store on first use."""
    with _lock:
        state = _load_state()
        vs_id = state.get("vector_store_id")
        if vs_id:
            try:
                client().vector_stores.retrieve(vs_id)
                return vs_id
            except NotFoundError:
                pass  # store was deleted remotely; create a fresh one
        store = client().vector_stores.create(name=VECTOR_STORE_NAME)
        state["vector_store_id"] = store.id
        _save_state(state)
        return store.id


# --------------------------------------------------------------------------
# Document management
# --------------------------------------------------------------------------

def upload_document(filename: str, content: bytes) -> dict[str, Any]:
    """Upload a file to OpenAI and index it into the vector store.

    Blocks until OpenAI finishes chunking/embedding so the caller knows the
    document is queryable (or failed) when this returns.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    vs_id = get_vector_store_id()
    file_obj = client().files.create(
        file=(filename, io.BytesIO(content)), purpose="assistants"
    )
    vs_file = client().vector_stores.files.create_and_poll(
        vector_store_id=vs_id, file_id=file_obj.id
    )
    if vs_file.status != "completed":
        err = getattr(vs_file.last_error, "message", None) or vs_file.status
        # Clean up the orphaned file so it doesn't linger in the account.
        try:
            client().files.delete(file_obj.id)
        except Exception:
            pass
        raise RuntimeError(f"Indexing failed for '{filename}': {err}")

    return {
        "file_id": file_obj.id,
        "filename": filename,
        "bytes": file_obj.bytes,
        "status": vs_file.status,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }


def list_documents() -> list[dict[str, Any]]:
    """List all documents currently indexed in the vector store."""
    vs_id = get_vector_store_id()
    docs = []
    for vs_file in client().vector_stores.files.list(
        vector_store_id=vs_id, limit=100
    ):
        try:
            meta = client().files.retrieve(vs_file.id)
            filename, size = meta.filename, meta.bytes
        except NotFoundError:
            filename, size = "(deleted file)", None
        docs.append(
            {
                "file_id": vs_file.id,
                "filename": filename,
                "bytes": size,
                "status": vs_file.status,
                "created_at": vs_file.created_at,
            }
        )
    docs.sort(key=lambda d: d["created_at"] or 0, reverse=True)
    return docs


def delete_document(file_id: str) -> None:
    """Remove a document from the vector store and delete the file."""
    vs_id = get_vector_store_id()
    try:
        client().vector_stores.files.delete(file_id, vector_store_id=vs_id)
    except NotFoundError:
        pass
    try:
        client().files.delete(file_id)
    except NotFoundError:
        pass


# --------------------------------------------------------------------------
# Question answering
# --------------------------------------------------------------------------


def ask(
    question: str,
    previous_response_id: str | None = None,
    provider: Provider = "openai",
) -> dict[str, Any]:
    """Answer a question grounded in the uploaded documents.

    Returns the answer text with inline [n] citation markers, the citation
    list, the retrieved source passages, and a response id that the client
    can pass back for conversational follow-ups.
    """
    # Determine whether this is a personal/sensitive request. IMPORTANT: we do not
    # short-circuit here — we still attempt to answer from uploaded documents first.
    # If no relevant doc match is found, we'll return the scripted refusal.
    refusal = _sensitive_internal_refusal(question)

    if _is_smalltalk(question):
        return {
            "answer": (
                "I’m doing well — thanks for asking. "
                "Upload a document any time, and I’ll help you find answers based on it."
            ),
            "citations": [],
            "sources": [],
            "response_id": previous_response_id,
        }

    if DEBUG:
        _log.info(
            "ask: q=%r web_enabled=%s",
            question,
            can_use_web_search(question),
        )

    # 1) Try internal document retrieval (OpenAI vector store). Works for both
    # providers if OPENAI_API_KEY is set.
    sources: list[dict[str, Any]] = []
    retrieval_resp_id: str | None = None
    try:
        if os.getenv("OPENAI_API_KEY"):
            sources, retrieval_resp_id = _retrieve_doc_sources(question, previous_response_id=previous_response_id)
    except Exception as exc:
        if DEBUG:
            _log.info("ask: retrieval failed: %s", exc)
        sources = []

    if DEBUG:
        top = sorted(
            ((s.get("filename") or s.get("file_id"), s.get("score")) for s in sources),
            key=lambda t: (t[1] is None, -(t[1] or 0)),
        )[:5]
        _log.info("ask: retrieved=%s top=%s", len(sources), top)

    # 1) Check whether the question has a relevant match in uploaded documents.
    matched_docs = _has_document_match(sources)
    if DEBUG:
        _log.info("ask: matched_docs=%s (min_relevance=%s)", matched_docs, MIN_DOC_RELEVANCE)

    # 2) If matched, answer from documents only (with citations).
    if matched_docs:
        context, citations = _build_numbered_passages(sources)
        prompt = (
            "Answer using ONLY the provided DOCUMENT PASSAGES. "
            "Cite every factual claim with [n] where n is the passage number. "
            "Return JSON only as {\"answer\": string, \"confidence\": number} (0..1).\n\n"
            f"Question: {question}\n\n{context}"
        )
        answer, _conf, resp_id = _llm_answer_json(provider, prompt, previous_response_id=previous_response_id)
        return {
            "answer": (answer or "").strip() or "(no answer)",
            "citations": citations,
            "sources": sources,
            "response_id": resp_id or retrieval_resp_id or previous_response_id,
        }

    # 2b) No doc match: if personal/sensitive, return the scripted refusal.
    if refusal is not None:
        if DEBUG:
            _log.info("ask: no doc match; returning scripted refusal")
        return {
            "answer": refusal,
            "citations": [],
            "sources": [],
            "response_id": retrieval_resp_id or previous_response_id,
        }

    # 3) No doc match: compare the model-knowledge answer with a Tavily-grounded answer.
    if _question_about_model(question):
        if provider == "openai":
            ans = f"In OpenAI mode, this app is configured to use `{CHAT_MODEL}`."
        else:
            try:
                ans = f"In LM Studio mode, this app is configured to use `{_resolve_lm_studio_model()}`."
            except Exception:
                ans = "In LM Studio mode, this app uses the model currently loaded in LM Studio."
        return {"answer": ans, "citations": [], "sources": [], "response_id": previous_response_id}

    prompt = (
        "Return JSON only as {\"answer\": string, \"confidence\": number} (0..1).\n\n"
        f"Question: {question}"
    )
    gen_answer, gen_conf, gen_id = _llm_answer_json(provider, prompt, previous_response_id=previous_response_id)
    if DEBUG:
        _log.info("ask: model-knowledge confidence=%s", gen_conf)

    if can_use_web_search(question):
        web = _ask_with_web_sources(question, provider, previous_response_id=previous_response_id)
        web_conf = float(web.get("confidence") or 0)
        if DEBUG:
            _log.info("ask: tavily-grounded confidence=%s", web_conf)
        if web_conf >= gen_conf:
            web_answer = (web.get("answer") or "").strip()
            prefix = (
                "I compared model knowledge with web sources and selected the web-grounded answer. "
                "Below is the answer based on the sources found:\n----\n"
            )
            web["answer"] = prefix + (web_answer or "(no answer)")
            return web

    # Tavily is disabled/unavailable, or the model-knowledge answer scored higher.
    return {
        "answer": (gen_answer or "").strip() or "(no answer)",
        "citations": [],
        "sources": [],
        "response_id": gen_id or previous_response_id,
    }
