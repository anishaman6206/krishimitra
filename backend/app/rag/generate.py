# backend/app/rag/generate.py
import time
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from app.config import settings

def t() -> float:
    return time.perf_counter()

# rough budget guard; adjust if your model/context window changes
_MAX_CONTEXT_CHARS = 700 * 8  # ~8 chunks * 700 chars each by default

def _compact_context(passages: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    total = 0
    for i, p in enumerate(passages or [], start=1):
        txt = (p.get("text") or "").strip().replace("\n", " ")
        if len(txt) > 700:
            txt = txt[:700] + " …"
        frag = f"[{i}] {txt}"
        if total + len(frag) > _MAX_CONTEXT_CHARS:
            break
        chunks.append(frag)
        total += len(frag)
    return "\n\n".join(chunks) if chunks else "—"

def _extract_week_bullets(tool_notes: str) -> str:
    if not tool_notes:
        return ""
    bullets: List[str] = []
    for ln in tool_notes.splitlines():
        tline = ln.strip()
        up = tline.upper()
        if up.startswith("WEEK 1:") or up.startswith("WEEK 2:"):
            bullets.append(f"• {tline}")
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

    try:
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        answer_text = getattr(resp, "content", str(resp)).strip()
    except Exception as e:
        return {"error": f"generation_failed: {e}", "answer": "", "sources": []}

    # Build distinct source list (max 5)
    sources: List[Dict[str, Any]] = []
    seen = set()
    for p in topk or []:
        key = (p.get("source"), p.get("page"))
        if key in seen:
            continue
        seen.add(key)
        title = p.get("title")
        if not title:
            src = (p.get("source") or "")
            title = src.rsplit("/", 1)[-1] or src
        sources.append({
            "title": title,
            "source": p.get("source"),
            "page": p.get("page"),
            "score": float(p.get("score") or 0.0),
        })
        if len(sources) >= 5:
            break

    total_ms = round((t() - start) * 1000)
    approx_tokens = (len(system) + len(user)) // 4
    print(f"⏱️  LLM synthesis: {total_ms}ms (~{approx_tokens} prompt tokens)")
    return {"answer": answer_text, "sources": sources}
