# backend/services/vector_store.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorStore(ABC):
    """Abstract interface for vector database providers.

    Both Pinecone and Supabase implement this interface.
    The RAG pipeline only talks to this — never to providers directly.
    """

    @abstractmethod
    def upsert(self, chunks: List[str], vectors: List[List[float]], doc_id: str) -> None:
        """Store text chunks and their vectors."""
        pass

    @abstractmethod
    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find the most similar chunks to a query vector.

        When ``doc_id`` is set, results are limited to chunks from that upload.

        Returns: list of dicts with 'text' and 'score' keys.
        """
        pass


def get_vector_store(provider: Optional[str] = None) -> VectorStore:
    """Factory: return the active vector store.

    ``provider`` overrides ``VECTOR_STORE`` from the environment when set
    (e.g. per-request Pinecone vs Supabase for parallel demos).
    """
    from config import settings

    name = (provider or settings.vector_store or "pinecone").strip().lower()
    if name == "pinecone":
        from services.pinecone_store import PineconeStore
        return PineconeStore()
    if name == "supabase":
        from services.supabase_store import SupabaseStore
        return SupabaseStore()
    raise ValueError(f"Unknown vector store: {name!r}")
