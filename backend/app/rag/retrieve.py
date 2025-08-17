import time
from typing import List, Dict, Any
from app.rag.index import build_or_load_index

def t() -> float:
    return time.perf_counter()

def retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    start = t()

    db = build_or_load_index(rebuild=False)
    results = db.similarity_search_with_relevance_scores(query, k=k)

    out: List[Dict[str, Any]] = []
    for doc, score in results:
        md = doc.metadata or {}
        out.append({
            "score": float(score),
            "text": doc.page_content,
            "source": md.get("source"),
            "title": md.get("title"),
            "page": md.get("page"),
        })

    total_ms = round((t() - start) * 1000)
    print(f"⏱️  RAG retrieve: {total_ms}ms ({len(out)} results)")

    return out
