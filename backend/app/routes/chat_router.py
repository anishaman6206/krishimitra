from fastapi import APIRouter, HTTPException
from ..schemas import ChatRequest, ChatResponse
from ..services.speech import transcribe
from ..rag import retrieve

router = APIRouter()

# basic intent detection - need to improve
def detect_intent(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in ["irrigate", "irrigation", "water"]): return "irrigation"
    if any(k in t for k in ["price", "sell", "wait", "market", "mandi"]): return "prices"
    if any(k in t for k in ["satellite", "ndvi", "field", "health"]): return "satellite"
    if any(k in t for k in ["disease", "leaf", "spot"]): return "disease"
    return "rag_fallback"

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.type == "voice":
        text = transcribe(req.content or req.media_url or "")
        intent = detect_intent(text)
    elif req.type == "text":
        text = req.content or ""
        intent = detect_intent(text)
    elif req.type == "image":
        text, intent = "", "disease"
    else:
        raise HTTPException(400, "Unsupported type")

    # 2) route
    # if intent == "disease":
    #     out = disease.predict(req.content or req.media_url or "")
    #     msg = f"{out['label']} (p={out['prob']:.2f}). Remedy: {out['remedy']}"
    #     return ChatResponse(text=msg, intent=intent, confidence=float(out.get("prob", 0.7)),
    #                         citations=out.get("sources", []), extras=out)

    # if intent == "satellite":
    #     aoi = req.metadata.get("aoi") or {"name": "Dharwad"}
    #     out = satellite.analyze(aoi)
    #     msg = f"Health score {out['health_score']:.2f}, moisture {out['moisture_proxy']}."
    #     return ChatResponse(text=msg, intent=intent, confidence=0.75,
    #                         citations=out.get("sources", []), extras=out)

    # if intent == "prices":
    #     commodity = req.metadata.get("commodity", "tomato")
    #     district = req.metadata.get("district", "Dharwad")
    #     out = yield_price.evaluate(commodity, district)
    #     msg = f"{commodity.title()} trend: {out['price_direction']}. Recommendation: {out['sell_wait']}."
    #     return ChatResponse(text=msg, intent=intent, confidence=float(out.get("confidence", 0.7)),
    #                         citations=out.get("sources", []), extras=out)

    # fallback: RAG stub
    ans, sources = retrieve.answer(text)
    return ChatResponse(text=ans, intent=intent, confidence=0.6, citations=sources, extras={})
