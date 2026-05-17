# backend/services/embeddings.py
from openai import OpenAI
from config import settings
from typing import List

client = OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of text chunks. Returns a list of vectors."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    return response.data[0].embedding
