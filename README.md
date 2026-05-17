# RAG Document Chat API

FastAPI service for Stage 2 RAG: upload PDFs, embed chunks into Pinecone or Supabase pgvector, and stream answers with retrieval-augmented generation.

## Requirements

- Python 3.11+ recommended
- API keys: Anthropic (chat), OpenAI (embeddings), Pinecone (when using Pinecone). For Supabase mode, also configure `SUPABASE_URL` and `SUPABASE_KEY` (or `SUPABASE_SERVICE_ROLE_KEY`).

## Setup

```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your secrets. Do not commit `.env`.

`config.py` requires `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and `PINECONE_API_KEY` at import time, even if you only call endpoints with `vector_store=supabase`.

## Run

From this directory (so `.env` loads correctly):

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- Health: `GET http://localhost:8000/health`
- OpenAPI docs: `http://localhost:8000/docs`

CORS is limited to `http://localhost:3000` for browser demos.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Claude for streamed RAG answers |
| `OPENAI_API_KEY` | Text embeddings |
| `PINECONE_API_KEY` | Pinecone client |
| `PINECONE_INDEX_NAME` | Index name (default `rag-documents`) |
| `SUPABASE_URL` | Supabase project URL (pgvector path) |
| `SUPABASE_KEY` or `SUPABASE_SERVICE_ROLE_KEY` | Service role key on the server only |
| `VECTOR_STORE` | Default backend: `pinecone` or `supabase` |

Per-request overrides: `POST /upload` and `POST /chat` accept `vector_store` (`pinecone` or `supabase`). Supabase returns 400 if URL/key are missing.

For Supabase schema and RPC, apply the SQL under `../supabase/` in this repo as needed.

## API

- **`POST /upload`** — multipart form: `file` (PDF only), optional `vector_store`. Returns `doc_id`, `filename`, `chunk` count. Image-only or scanned PDFs with no extractable text are rejected with 400.
- **`POST /chat`** — JSON body: `{ "question": "...", "doc_id": "<uuid from upload>", "vector_store": null }`. Response is `text/plain` streaming.

## Project layout

- `main.py` — routes and CORS
- `config.py` — settings from environment
- `services/` — PDF extraction, chunking, embeddings, vector stores, RAG streaming
