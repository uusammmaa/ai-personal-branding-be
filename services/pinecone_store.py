# backend/services/pinecone_store.py
from pinecone import Pinecone
from services.vector_store import VectorStore
from config import settings
from typing import List, Dict, Any, Optional


class PineconeStore(VectorStore):
    def __init__(self):
        pc = Pinecone(api_key=settings.pinecone_api_key)
        self.index = pc.Index(settings.pinecone_index_name)

    def upsert(self, chunks: List[str], vectors: List[List[float]], doc_id: str) -> None:
        """Store chunks in Pinecone with metadata."""
        records = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            records.append(
                {
                    "id": f"{doc_id}-{i}",
                    "values": vector,
                    "metadata": {"text": chunk, "doc_id": doc_id},
                }
            )
        batch_size = 100
        for i in range(0, len(records), batch_size):
            self.index.upsert(vectors=records[i : i + batch_size])

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search Pinecone for similar chunks."""
        kwargs: Dict[str, Any] = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": True,
        }
        if doc_id:
            kwargs["filter"] = {"doc_id": doc_id}
        result = self.index.query(**kwargs)
        return [
            {"text": match.metadata["text"], "score": match.score}
            for match in result.matches
        ]
