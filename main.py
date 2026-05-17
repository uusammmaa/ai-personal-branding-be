# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.document import extract_text_from_pdf, chunk_text
from services.embeddings import embed_texts
from services.vector_store import get_vector_store
from services.rag import rag_stream
from config import settings
import uuid

app = FastAPI(title="RAG Document Chat API")

UNSUPPORTED_PDF_NO_TEXT = (
    "This PDF type is not supported: no extractable text was found. "
    "That usually means the file is a screenshot, a scan, or a print-to-PDF "
    "page saved as images rather than real text. "
    "Only text-based PDFs work here—export from Word/Google Docs, or use a PDF "
    "where you can select and copy text in a reader."
)


def _resolve_vector_store(requested: str | None) -> str:
    name = (requested or settings.vector_store or "pinecone").strip().lower()
    if name not in ("pinecone", "supabase"):
        raise HTTPException(
            status_code=400,
            detail="vector_store must be 'pinecone' or 'supabase'",
        )
    if name == "supabase" and (
        not settings.supabase_url.strip() or not settings.supabase_key.strip()
    ):
        raise HTTPException(
            status_code=400,
            detail="Supabase is not configured (set SUPABASE_URL and SUPABASE_KEY or SUPABASE_SERVICE_ROLE_KEY)",
        )
    return name


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    vector_store: str | None = Form(None),
):
    """Upload a PDF, chunk it, embed it, store in vector DB."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    store_name = _resolve_vector_store(vector_store)

    file_bytes = await file.read()
    doc_id = str(uuid.uuid4())

    # Process document
    text = extract_text_from_pdf(file_bytes)
    chunks = chunk_text(text)
    if not any(chunk.strip() for chunk in chunks):
        raise HTTPException(status_code=400, detail=UNSUPPORTED_PDF_NO_TEXT)
    vectors = embed_texts(chunks)

    # Store in vector DB
    store = get_vector_store(store_name)
    store.upsert(chunks, vectors, doc_id)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks": len(chunks)
    }


class ChatRequest(BaseModel):
    question: str
    doc_id: str
    vector_store: str | None = None


@app.post("/chat")
async def chat(request: ChatRequest):
    """RAG chat: embed question → retrieve → stream Claude answer."""
    doc_id = request.doc_id.strip()
    if not doc_id:
        raise HTTPException(
            status_code=400,
            detail="doc_id is required — upload a document first",
        )
    store_name = _resolve_vector_store(request.vector_store)
    return StreamingResponse(
        rag_stream(request.question, doc_id, vector_store=store_name),
        media_type="text/plain",
    )
