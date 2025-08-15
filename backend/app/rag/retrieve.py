from typing import List, Dict, Any
from app.rag.index import build_or_load_index

def retrieve(query: str, k: int = 4) -> List[Dict[str, Any]]:
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
    return out
