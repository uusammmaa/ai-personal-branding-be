# backend/services/supabase_store.py
from typing import Any, Dict, List, Optional

from supabase import create_client

from config import settings
from services.vector_store import VectorStore


class SupabaseStore(VectorStore):
    def __init__(self) -> None:
        self.client = create_client(settings.supabase_url, settings.supabase_key)

    def upsert(self, chunks: List[str], vectors: List[List[float]], doc_id: str) -> None:
        records = [
            {
                "id": f"{doc_id}-{i}",
                "doc_id": doc_id,
                "content": chunk,
                "embedding": vector,
            }
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        batch_size = 100
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            self.client.table("documents").insert(batch).execute()

    def query(
        self,
        vector: List[float],
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not doc_id:
            return []
        result = self.client.rpc(
            "match_documents",
            {
                "query_embedding": vector,
                "match_count": top_k,
                "filter_doc_id": doc_id,
            },
        ).execute()
        rows = result.data or []
        return [
            {"text": row["content"], "score": float(row["similarity"])}
            for row in rows
        ]
