# 🧠 DocIntel AI

### AI-Powered Intelligent Document Analysis & Retrieval Platform

DocIntel AI is a Retrieval-Augmented Generation (RAG) platform for chatting with, semantically searching, and summarizing your documents. Upload PDFs, Word documents, plain text, or Markdown files, and DocIntel AI turns them into a queryable knowledge base — every answer is grounded in your documents and comes with citations back to the exact page and excerpt it was drawn from.

Built as a production-quality AI engineering capstone: clean layered architecture, dependency injection throughout, and 300+ automated tests across both the Python backend and the API layer (service-layer mocks, FastAPI `TestClient` route tests). The React frontend (`frontend/`) is the sole UI, served together with the API by `api/main.py`, both built on the same service layer underneath.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Folder Structure](#folder-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running the Tests](#running-the-tests)
- [Deployment](#deployment)
- [Screenshots](#screenshots)
- [Future Roadmap](#future-roadmap)
- [License](#license)

---

## Features

| Feature | Description |
|---|---|
| **Document Upload** | Drag-and-drop, multi-file, per-file progress, duplicate detection (SHA-256 content hash), validation feedback |
| **Document Library** | Search, sort, status badges, chunk counts, preview, delete-with-confirmation |
| **AI Chat** | Streaming responses, ChatGPT-style UI, citations with page numbers and match scores, regenerate, copy, clear conversation, scope chat to specific documents |
| **Semantic Search** | Standalone similarity search (no LLM call) with highlighted matching terms and ranked results |
| **AI Summary** | Executive summary, key insights, and topics per document — with automatic map-reduce summarization for documents too long for a single LLM context |
| **Citations** | Every AI answer traces back to document, page, chunk, and similarity score |
| **Analytics Dashboard** | Document/chunk/embedding counts, average response time, storage usage, most-queried documents, upload activity chart |
| **Settings** | Chunk size/overlap, retrieval Top-K, temperature, max tokens, model selection — genuinely functional (threaded through as per-call overrides, not decorative) |
| **Export** | Download conversations as PDF or Markdown, copy full transcript |

---

## Architecture

```
                         ┌─────────────────────┐
                         │      React UI          │
                         │  (chat / search /      │
                         │   upload / analytics)  │
                         └──────────┬────────────┘
                                    │
                         ┌──────────▼────────────┐
                         │      FastAPI            │
                         │  (routers / schemas)    │
                         └──────────┬────────────┘
                                    │
                         ┌──────────▼────────────┐
                         │    Service Layer        │
                         │  DocumentService        │
                         │  ChatService             │
                         │  SearchService           │
                         │  SummaryService          │
                         │  AnalyticsService        │
                         │  SettingsService         │
                         │  ExportService           │
                         └───┬──────────────┬──────┘
                             │              │
              ┌──────────────▼───┐   ┌─────▼──────────────┐
              │ Document Pipeline │   │   RAG Pipeline       │
              │ loaders/          │   │ Retriever            │
              │ preprocessing/    │   │ PromptBuilder         │
              │ (clean, chunk)    │   │ BaseLLM → OllamaCloud │
              └─────────┬─────────┘   └──────────┬───────────┘
                        │                          │
              ┌─────────▼─────────┐     ┌─────────▼─────────┐
              │  EmbeddingService  │────▶│    ChromaManager    │
              │ (Sentence-         │     │   (ChromaDB vector  │
              │  Transformers)     │     │    store)           │
              └────────────────────┘     └─────────────────────┘
                        │                          │
                        └────────────┬─────────────┘
                                     │
                          ┌──────────▼───────────┐
                          │   SQLiteManager        │
                          │ (documents, chunks,    │
                          │  chat, analytics)      │
                          └────────────────────────┘
```

**Design principles:**
- **UI never touches the database or business logic directly** — the React frontend (`frontend/`) only ever calls the FastAPI routes in `api/routers/`, which call a service, nothing else.
- **Dependency injection everywhere** — every service accepts its collaborators (DB, embedding model, LLM, vector store) as constructor arguments with sensible defaults. This is what makes hundreds of tests possible without a single real network call: fakes are injected at exactly this seam.
- **Repository pattern for persistence** — `SQLiteManager` is the only module that writes raw SQL. `ChromaManager` is the only module that imports `chromadb` directly.
- **Every exception path lands as a clean error message, never a raw traceback** — a mapped HTTP status + JSON envelope, built from the `DocIntelError` hierarchy.

### RAG Pipeline Flow

```
User Question
   → Retriever.retrieve()          [embed query, similarity search, threshold filter]
   → PromptBuilder.build()         [inject context + citation markers]
   → BaseLLM.generate()/.stream()  [Ollama Cloud]
   → RAGResponse                   [answer + citations + timing]
```

If retrieval returns nothing above the similarity threshold, the LLM call is **skipped entirely** — a deterministic "no relevant information found" response is returned instead, avoiding both cost and hallucination risk.

---

## Folder Structure

```
DocIntelAI/
├── config.py                   # Centralized settings (reads .env)
├── requirements.txt
├── render.yaml                  # Render Blueprint — FastAPI + React
├── .env.example
│
├── api/                          # FastAPI backend
│   ├── main.py                    # App factory: CORS, error handlers, router + static registration
│   ├── errors.py                   # DocIntelError -> HTTP status mapping
│   ├── dependencies.py              # DI providers for service singletons
│   ├── static.py                     # Serves frontend/dist/ with SPA fallback in production
│   ├── schemas/                       # Pydantic request/response models, one file per resource
│   └── routers/                        # documents, chat, search, summary, analytics, settings, export, health
│
├── frontend/                     # Vite + vanilla JS + vanilla CSS frontend
│   ├── src/
│   │   ├── main.js  app.js  router.js
│   │   ├── api/                       # client.js (fetch/SSE wrapper), resources.js (per-endpoint calls)
│   │   ├── components/                 # sidebar, modal, toast, documentScopeSelect
│   │   ├── pages/                       # chat, search, documents, analytics
│   │   └── styles/                       # tokens/base/components
│   └── vite.config.js                # Dev-server proxy to the FastAPI backend, no CORS needed locally
│
├── docs/
│   └── DEPLOYMENT.md             # Full Render deployment guide
├── uploads/                       # Uploaded files (gitignored)
├── database/                      # SQLite database (gitignored)
├── vectorstore/                   # ChromaDB persistence (gitignored)
├── logs/                          # Rotating log files (gitignored)
│
├── tests/                          # 300+ tests, mirrors src/ and api/ structure
│   ├── conftest.py                  # Shared fakes (FakeEmbeddingModel, FakeOllamaChatClient, ...)
│   ├── test_security.py             # Path traversal, XSS, secret-leakage regression tests
│   ├── loaders/  preprocessing/  embeddings/  vectorstore/
│   ├── rag/  llm/  database/  services/
│   └── api/                           # FastAPI TestClient route tests, one file per resource
│
└── src/
    ├── loaders/                     # PDF (PyMuPDF), DOCX (python-docx), TXT/MD
    │   ├── base_loader.py             # Abstract interface + LoadedDocument
    │   ├── pdf_loader.py  docx_loader.py  txt_loader.py
    │   └── loader_factory.py           # Extension → loader registry
    │
    ├── preprocessing/
    │   ├── cleaner.py                  # Unicode/whitespace/ligature normalization
    │   └── splitter.py                 # RecursiveCharacterTextSplitter wrapper
    │
    ├── embeddings/
    │   └── embedding_service.py         # BAAI/bge-base-en-v1.5, lazy-loaded, batched
    │
    ├── vectorstore/
    │   └── chroma_manager.py            # ChromaDB CRUD + similarity query
    │
    ├── rag/
    │   ├── retriever.py  prompt_builder.py  pipeline.py
    │
    ├── llm/
    │   ├── base.py                      # Abstract BaseLLM interface
    │   └── ollama_cloud.py               # Ollama Cloud implementation via the official SDK (streaming + non-streaming)
    │
    ├── database/
    │   ├── models.py                     # Typed dataclasses (Document, Chunk, ChatMessage, ...)
    │   └── sqlite_manager.py              # Repository pattern, all SQL lives here
    │
    ├── utils/
    │   ├── helpers.py  logger.py  exceptions.py
    │
    └── services/
        ├── document_service.py  chat_service.py  search_service.py
        ├── summary_service.py  analytics_service.py  settings_service.py
        └── export_service.py
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vite + vanilla JavaScript + vanilla CSS (`frontend/`) |
| API | FastAPI, serving both REST/SSE routes and the built frontend from one process (`api/`) |
| AI Framework | LangChain (text splitting) |
| LLM | Ollama Cloud API (via the official `ollama` Python SDK) — `gpt-oss:120b-cloud` |
| Embeddings | Sentence-Transformers — `BAAI/bge-base-en-v1.5` |
| Vector Database | ChromaDB |
| Metadata Database | SQLite |
| Document Parsing | PyMuPDF, python-docx |
| PDF Export | ReportLab |
| Testing | pytest, FastAPI `TestClient` |
| Deployment | Render |

---

## Installation

### Prerequisites
- Python 3.12
- Node.js + npm (for the frontend)
- An [Ollama Cloud](https://ollama.com) API key

### Setup

```bash
# Clone and enter the project
git clone <your-repo-url>
cd DocIntelAI

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install backend dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and set OLLAMA_CLOUD_API_KEY

# Terminal 1 — run the API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — run the frontend dev server
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (typically `http://localhost:5173`) — its dev server proxies `/api/*` to the FastAPI backend, so no CORS setup is needed locally. See `frontend/README.md` for more on the frontend's structure.

On first document upload, the embedding model (`BAAI/bge-base-en-v1.5`, ~430MB) downloads automatically from Hugging Face — this happens once and is cached locally.

---

## Environment Variables

All variables are documented in [`.env.example`](.env.example) with working defaults in `config.py`. The only one you must set is:

| Variable | Description |
|---|---|
| `OLLAMA_CLOUD_API_KEY` | Your Ollama Cloud API key — required for chat/summarization to work |

Everything else (chunk size, retrieval Top-K, temperature, model name, upload limits, etc.) has a sensible default and can also be changed at runtime from the app's **Settings** panel without touching environment variables at all.

---

## Running the Tests

```bash
pip install -r requirements.txt
pytest
```

300+ tests across every layer — loaders, cleaning/splitting, embeddings, vector store, retrieval, prompt building, LLM client (error paths: timeout, auth, malformed responses, streaming), all seven services, and the FastAPI layer (every route, the `DocIntelError` → HTTP status mapping, SSE streaming, static-frontend serving) — with zero real network calls (fakes injected at the embedding-model, LLM-client, and service-dependency boundaries) and zero real LLM API costs.

```bash
# Run a specific layer
pytest tests/services/ -v
pytest tests/api/ -v
pytest tests/test_security.py -v
```

---

## Deployment

The stack is **FastAPI + React** (`api/` + `frontend/`, one process serving both) — push to GitHub, create a Render Web Service with the build/start commands from `docs/DEPLOYMENT.md`, set `OLLAMA_CLOUD_API_KEY`, deploy. Full step-by-step instructions, including the persistent-disk setup needed for uploaded documents and chat history to survive restarts, are in **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

A `render.yaml` Blueprint is included, ready to deploy as-is.

---

## Screenshots

> _Add screenshots of the Chat, Search, Documents, and Analytics pages here once deployed._

```
[ Chat page — streaming answer with citation panel ]
[ Semantic Search — highlighted matches with similarity scores ]
[ Document Library — upload, preview, AI summary ]
[ Analytics Dashboard — usage metrics and charts ]
```

---

## Future Roadmap

The React frontend (`frontend/`) is the UI, served together with the API by `api/main.py`. From here, designed so each of these can be added without touching the core RAG pipeline:

- **PostgreSQL** — swap `SQLiteManager` for a `PostgresManager` implementing the same method signatures
- **Redis** — cache embedding lookups and session state for multi-instance deployments
- **Authentication** — multi-user support with per-user document collections (the `Collection` model and schema already exist, just unused by the UI)
- **Cloud Storage** — swap local `uploads/` for S3/GCS behind the same `DocumentService` interface
- **Persisted summaries** — a `summaries` table so AI summaries survive across sessions instead of being regenerated

---

## License

MIT — see [LICENSE](LICENSE).
