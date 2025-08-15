# backend/app/rag/generate.py
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from app.config import settings

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
    bullets = []
    for ln in tool_notes.splitlines():
        t = ln.strip()
        if t.upper().startswith("WEEK 1:") or t.upper().startswith("WEEK 2:"):
            bullets.append(f"• {t}")
    return "\n".join(bullets)

def synthesize(question: str, topk: List[Dict[str, Any]], tool_notes: str = "") -> Dict[str, Any]:
    llm = ChatOpenAI(model=settings.OPENAI_MODEL, temperature=0.3, timeout=30)

    context_block = _compact_context(topk)
    week_bullets = _extract_week_bullets(tool_notes)

    system = (
        "You are KrishiMitra, a practical farm advisor for India. "
        "Decide using the numeric facts provided. Keep answers short, simple, and useful. "
        "Avoid technical jargon (no quantiles, p50/p80, embeddings). "
        "If some facts are missing, be upfront and give a cautious, practical next step."
    )

    user = (
        f"Question:\n{question}\n\n"
        f"Context (may help for agronomy/policy guidance):\n{context_block}\n\n"
        f"{tool_notes or '—'}\n\n"
        "TASK:\n"
        "- Read the FACTS_JSON (if present) and use those numbers first to make a decision.\n"
        "- Start with one clear recommendation (e.g., 'Sell now…', 'Wait one week…', 'Irrigate lightly tomorrow…').\n"
        "- If WEEK 1 / WEEK 2 lines are present, list them as simple bullets under 'Forecast'.\n"
        "- Keep it farmer-friendly. 3–8 short sentences plus the bullets.\n"
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

    return {"answer": answer_text, "sources": sources}
