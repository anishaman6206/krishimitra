# backend/app/rag/generate.py
import time
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from app.config import settings

def t(): return time.perf_counter()

def _compact_context(passages: List[Dict[str, Any]]) -> str:
    chunks = []
    for i, p in enumerate(passages, start=1):
        txt = (p.get("text") or "").strip().replace("\n", " ")
        if len(txt) > 700:
            txt = txt[:700] + " …"
        chunks.append(f"[{i}] {txt}")
    return "\n\n".join(chunks) if chunks else "—"



def _extract_week_bullets(tool_notes: str) -> str:
    if not tool_notes:
        return ""
    tool_notes = tool_notes[:6000] + " …"
    bullets = []
    for ln in tool_notes.splitlines():
        t = ln.strip()
        if t.upper().startswith("WEEK 1:") or t.upper().startswith("WEEK 2:"):
            bullets.append(f"• {t}")
    return "\n".join(bullets)

def synthesize(question: str, topk: List[Dict[str, Any]], tool_notes: str = "") -> Dict[str, Any]:
    start = t()
    llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0.3, timeout=30)

    context_block = _compact_context(topk)
    week_bullets = _extract_week_bullets(tool_notes)

    system = (
        "You are KrishiMitra, a practical farm advisor for India. "
        "Use ONLY the provided context and tool notes. "
        "If a tool note is unrelated to the user's question, IGNORE it. "
        "Decide using numeric facts first. Keep answers short and practical. "
        "Avoid technical jargon (no p50/p80). Do NOT include sources in the answer."
    )
    
    user = (
        f"Question:\n{question}\n\n"
        f"Context passages:\n{context_block}\n\n"
        "Relevance rules:\n"
        "- Use price/forecast notes only for market/sell/wait/rate queries.\n"
        "- Use NDVI/NDMI/NDWI/LAI only for crop condition/irrigation/stress queries.\n"
        "- Use weather only for weather/irrigation timing/heat-cold-rain risk.\n\n"
        f"{tool_notes or '—'}\n\n"
        "TASK:\n"
        "- Start with one clear recommendation.\n"
        
        "- Keep it farmer-friendly. 3–6 short sentences plus bullets.\n"
        "- Do not include citations or source names."
    )

    resp = llm.invoke([{"role": "system", "content": system},
                       {"role": "user", "content": user}])
    answer_text = resp.content.strip() if hasattr(resp, "content") else str(resp)

    # Return sources separately (not printed in the answer)
    sources = []
    seen = set()
    for p in topk:
        key = (p.get("source"), p.get("page"))
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "title": p.get("title") or (p.get("source") or "").split(".pdf")[0].title(),
            "source": p.get("source"),
            "page": p.get("page"),
            "score": float(p.get("score") or 0.0),
        })
        if len(sources) >= 5:
            break

    total_ms = round((t() - start) * 1000)
    tokens_est = len(system + user) // 4
    print(f"⏱️  LLM synthesis: {total_ms}ms (~{tokens_est} tokens)")

    return {"answer": answer_text, "sources": sources}
