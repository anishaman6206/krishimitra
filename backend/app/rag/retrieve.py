# backend/app/rag/retrieve.py
import time
from typing import List, Dict, Any, Tuple

from app.rag.index import build_or_load_index
from langchain.schema import Document  # for clarity of types

def t() -> float: 
    return time.perf_counter()

def _row(doc: Document, score: float) -> Dict[str, Any]:
    md = doc.metadata or {}
    return {
        "score": float(score),
        "text": doc.page_content or "",
        "source": md.get("source"),
        "title": md.get("title"),
        "page": md.get("page"),
    }

def retrieve(query: str, k: int = 4) -> List[Dict[str, Any]]:
    start = t()
    k = max(1, int(k))
    db = build_or_load_index(rebuild=False)

    # similarity_search_with_relevance_scores returns List[Tuple[Document, float]]
    results: List[Tuple[Document, float]] = db.similarity_search_with_relevance_scores(query, k=k) or []

    out: List[Dict[str, Any]] = [_row(doc, score) for doc, score in results if doc]
    total_ms = round((t() - start) * 1000)
    print(f"⏱️  RAG retrieve: {total_ms}ms ({len(out)} results)")
    return out
