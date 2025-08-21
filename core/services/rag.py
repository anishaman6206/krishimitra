# core/services/rag.py
from typing import List, Dict, Any, Optional
import asyncio

from backend.app.rag.retrieve import retrieve  # sync function

class RAGService:
    """Wrapper around existing RAG functionality."""

    def __init__(self, default_k: int = 4) -> None:
        self.default_k = max(1, int(default_k))

    async def retrieve_sources(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Run semantic retrieval in a worker thread to avoid blocking the event loop.
        Returns a list of {score, text, source, title, page}.
        """
        try:
            kk = max(1, int(k or self.default_k))
        except Exception:
            kk = self.default_k

        try:
            return await asyncio.to_thread(lambda: retrieve(query, k=kk)) or []
        except Exception as e:
            print(f"RAG error: {e}")
            return []
