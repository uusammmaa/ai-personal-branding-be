# backend/services/rag.py
from anthropic import Anthropic
from config import settings
from services.embeddings import embed_query
from services.vector_store import get_vector_store
from typing import Iterator

client = Anthropic(api_key=settings.anthropic_api_key)


def build_context_prompt(chunks: list[dict], question: str) -> str:
    """Build the prompt that injects retrieved chunks into Claude's context."""
    context = "\n\n---\n\n".join([c["text"] for c in chunks])
    return f"""You are a helpful assistant that answers questions based on provided document context.

Use ONLY the context below to answer the question. If the answer is not in the context, say so clearly.
Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""


def rag_stream(
    question: str,
    doc_id: str,
    vector_store: str | None = None,
) -> Iterator[str]:
    """Full RAG pipeline: embed → retrieve → prompt → stream."""
    # 1. Embed the question
    query_vector = embed_query(question)

    # 2. Retrieve top-5 relevant chunks (scoped to this upload only)
    store = get_vector_store(vector_store)
    chunks = store.query(query_vector, top_k=5, doc_id=doc_id)

    # 3. Build prompt with retrieved context
    prompt = build_context_prompt(chunks, question)

    # 4. Stream Claude's answer
    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            yield text
