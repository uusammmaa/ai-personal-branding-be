# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.document import extract_text_from_pdf, chunk_text
from services.embeddings import embed_texts
from services.vector_store import get_vector_store
from services.rag import rag_stream
import uuid

app = FastAPI(title="RAG Document Chat API")

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
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF, chunk it, embed it, store in vector DB."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    file_bytes = await file.read()
    doc_id = str(uuid.uuid4())

    # Process document
    text = extract_text_from_pdf(file_bytes)
    chunks = chunk_text(text)
    vectors = embed_texts(chunks)

    # Store in vector DB
    store = get_vector_store()
    store.upsert(chunks, vectors, doc_id)

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "chunks": len(chunks)
    }


class ChatRequest(BaseModel):
    question: str
    doc_id: str


@app.post("/chat")
async def chat(request: ChatRequest):
    """RAG chat: embed question → retrieve → stream Claude answer."""
    doc_id = request.doc_id.strip()
    if not doc_id:
        raise HTTPException(
            status_code=400,
            detail="doc_id is required — upload a document first",
        )
    return StreamingResponse(
        rag_stream(request.question, doc_id),
        media_type="text/plain"
    )
