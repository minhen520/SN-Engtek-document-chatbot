# Smith&Nephew + Engtek - Document Chatbot (RAG)

A retrieval-augmented generation app where the **document RAG pipeline runs on the OpenAI API** — no local embedding math, no third-party vector database.

Optionally, you can enable **Tavily web search** for *general* questions when your uploaded documents don’t contain an answer.

| Stage | How it's done |
|---|---|
| Chunking & embedding | OpenAI **Vector Stores API** (automatic on upload) |
| Vector storage | OpenAI-hosted vector store |
| Retrieval | OpenAI hybrid semantic + keyword search with reranking |
| Answer generation | OpenAI **Responses API** with the `file_search` tool |
| Citations | Native `file_citation` annotations, anchored to the answer text |

Users upload documents (PDF, DOCX, PPTX, TXT, MD, code files…), ask questions in a chat UI, and get grounded answers with numbered `[n]` citations plus the actual retrieved passages for verification.

## Quick start

```powershell
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Configure your API key
copy .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...

# 3. Run
uvicorn server:app --reload

# 4. Open http://localhost:8000
```

## Local LLM (LM Studio)

The UI includes a **Provider** dropdown:

- **OpenAI (API key)**: full document RAG (upload, search, citations)
- **LM Studio (local)**: chat-only via LM Studio's OpenAI-compatible server

To use LM Studio:

1. In LM Studio, load a model and start the **OpenAI-compatible server**.
2. Set in your `.env`:

	- `LM_STUDIO_URL=http://127.0.0.1:1234/v1`
	- `LM_STUDIO_MODEL=<your loaded model id>` (optional; if omitted, the server's first model is used)

Note: LM Studio mode does **not** support document upload/indexing/retrieval because this project’s RAG pipeline relies on OpenAI Vector Stores + the Responses `file_search` tool.

## How it works

1. **Upload** — files go to OpenAI Files (`purpose="assistants"`), then are attached to a persistent vector store via `vector_stores.files.create_and_poll`. OpenAI chunks (~800-token chunks, 400 overlap by default), embeds, and indexes them. The upload endpoint blocks until indexing completes, so a successful upload is immediately queryable.
2. **Ask** — `responses.create` is called with the `file_search` tool pointed at the vector store. The model decides what to search, OpenAI retrieves the top passages (hybrid search + reranking), and the model writes an answer grounded in them.
3. **Citations** — the response carries `file_citation` annotations with exact character offsets. The backend converts them into inline `[n]` markers and a citation list; the raw retrieved passages (with relevance scores) are returned too and shown under "Retrieved passages".
4. **Follow-ups** — the client passes `previous_response_id` back on each turn, so OpenAI threads the conversation server-side and follow-up questions ("what about section 3?") work naturally.
5. **Persistence** — the vector store id is saved in `data/state.json`, so documents survive restarts. Delete a document in the UI to remove it from both the vector store and OpenAI file storage.

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/upload` | Multipart upload, one or more files; waits for indexing |
| `GET` | `/api/documents` | List indexed documents |
| `DELETE` | `/api/documents/{file_id}` | Remove a document |
| `POST` | `/api/ask` | `{question, previous_response_id?}` → answer, citations, sources, response_id |
| `GET` | `/api/health` | Key/model check |

## Configuration (env vars)

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `RAG_CHAT_MODEL` | `gpt-4o-mini` | Generation model (try `gpt-4.1` or `o4-mini` for harder questions) |
| `RAG_VECTOR_STORE_NAME` | `rag-app-store` | Name used when creating the vector store |
| `RAG_MAX_RESULTS` | `8` | Max retrieved chunks per question |
| `RAG_WEB_SEARCH_ENABLED` | `0` | When `1`, allows web search fallback for general questions |
| `TAVILY_API_KEY` | — | Tavily API key (required for web search) |
| `RAG_TAVILY_MAX_RESULTS` | `5` | Max web results to fetch per question |
| `LM_STUDIO_URL` | `http://127.0.0.1:1234/v1` | LM Studio OpenAI-compatible base URL |
| `LM_STUDIO_MODEL` | — | LM Studio model id (optional; auto-picks first model if unset) |

## Grounding behavior

The system prompt instructs the model to answer **only** from retrieved passages, cite every claim, surface conflicts between documents, and say "the documents don't contain this" rather than guess.

Web search (optional):
- If `RAG_WEB_SEARCH_ENABLED=1` and `TAVILY_API_KEY` is set, the app may use Tavily for **general questions** when there are no relevant document passages.
- If no documents are uploaded and web search is not enabled, `/api/ask` returns a clear 400 error.

Refusal behavior:
- Personal HR/account-specific questions (e.g. leave balance, payroll) and internal operational/financial questions (e.g. equipment cost/budgets/invoices) are refused with guidance to contact the appropriate team.

## Notes & limits

- Per-file upload limit is 50 MB (configurable in `server.py`); OpenAI's own limit is 512 MB / ~5M tokens per file.
- Vector store storage is billed by OpenAI at $0.10/GB/day after the first free GB; the store persists until you delete it.
- `data/state.json` only stores the vector store id — your documents live in your OpenAI account, scoped to your API key.


Contributer Credit:
1. Toh Seong Thye (Smith & Nephew)
2. Muhamad Eizham (Smith & Nephew)
3. Khoo Jia Hen (Eng Teknologi Sdn Bhd)