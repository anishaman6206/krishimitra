"""
/ask endpoint
"""
from fastapi import APIRouter, Depends
from app.schemas import AskRequest, AskResponse
from app.di import get_conversation_service
from core.services.conversation import ConversationService
from core.models.io import ChannelMessageIn

router = APIRouter(tags=["ai"])

@router.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, 
              conversation_service: ConversationService = Depends(get_conversation_service)):
    """
    AI endpoint using the new conversation service.
    This now goes through the same brain as Telegram.
    """
    
    # Convert AskRequest to ChannelMessageIn
    geo = None
    if req.geo:
        geo = {"lat": req.geo.lat, "lon": req.geo.lon}
    
    channel_msg = ChannelMessageIn(
        channel="web",
        user_id="web_user",  # Could be session ID or user ID
        text=req.text,
        lang_hint=req.lang,
        geo=geo,
        metadata={
            "crop": req.crop,
            "state": req.state,
            "district": req.district,
            "market": req.market,
            "variety": req.variety,
            "grade": req.grade,
            "horizon_days": req.horizon_days,
            "qty_qtl": req.qty_qtl,
            "debug": req.debug
        }
    )
    
    # Use conversation service
    result = await conversation_service.handle(channel_msg)
    
    # Convert back to AskResponse format
    # Extract sources from tool_notes if available
    sources = []
    if result.tool_notes and "rag" in result.tool_notes:
        rag_data = result.tool_notes["rag"]
        if isinstance(rag_data, list):
            sources = rag_data
    
    return AskResponse(
        answer=result.text or "",
        sources=sources,
        tool_notes=result.tool_notes or {},
        follow_ups=result.followups,
        lang=req.lang or "en"
    )
