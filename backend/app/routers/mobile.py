"""
Mobile-specific API endpoints
"""
from fastapi import APIRouter, Depends
from app.schemas import AskRequest, AskResponse
from app.di import get_conversation_service
from core.services.conversation import ConversationService
from core.models.io import ChannelMessageIn
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

router = APIRouter(tags=["mobile"], prefix="/mobile")

class MobileAskRequest(BaseModel):
    text: str
    lang: Optional[str] = "hi"
    debug: bool = True
    lat: Optional[float] = None
    lon: Optional[float] = None

class MobileCard(BaseModel):
    kind: str  # 'price', 'weather', 'ndvi'
    data: Dict[str, Any]

class MobileAskResponse(BaseModel):
    answer: str
    cards: List[MobileCard] = []
    follow_ups: List[str] = []
    lang: str = "hi"

@router.post("/ask", response_model=MobileAskResponse)
async def mobile_ask(req: MobileAskRequest, 
                    conversation_service: ConversationService = Depends(get_conversation_service)):
    """
    Mobile-optimized AI endpoint with structured card responses.
    """
    
    # Convert to ChannelMessageIn
    geo = None
    if req.lat is not None and req.lon is not None:
        geo = {"lat": req.lat, "lon": req.lon}
    
    channel_msg = ChannelMessageIn(
        channel="web",  # Use 'web' since 'mobile' is not in the allowed channels
        user_id="mobile_demo_user",  # Demo user
        text=req.text,
        lang_hint=req.lang,
        geo=geo,
        metadata={
            "debug": req.debug,
            "source": "mobile_app"  # Add this to distinguish from regular web
        }
    )
    
    # Use conversation service
    result = await conversation_service.handle(channel_msg)
    
    # Extract structured cards from tool_notes
    cards = []
    if result.tool_notes:
        # Price card from price tool
        if "price" in result.tool_notes:
            price_data = result.tool_notes["price"]
            if isinstance(price_data, dict):
                cards.append(MobileCard(
                    kind="price",
                    data={
                        "commodity": price_data.get("commodity", "Unknown"),
                        "market": price_data.get("market", "Unknown"),
                        "district": price_data.get("district", "Unknown"),
                        "state": price_data.get("state", "Unknown"),
                        "modal": price_data.get("modal_price_inr_per_qtl", price_data.get("modal_price", 0)),
                        "min": price_data.get("min_price_inr_per_qtl", price_data.get("min_price", 0)),
                        "max": price_data.get("max_price_inr_per_qtl", price_data.get("max_price", 0)),
                        "date": price_data.get("arrival_date", "Unknown")
                    }
                ))
        
        # Weather card
        if "weather" in result.tool_notes:
            weather_data = result.tool_notes["weather"]
            if isinstance(weather_data, dict):
                cards.append(MobileCard(
                    kind="weather",
                    data={
                        "rain": weather_data.get("total_rain_next_24h_mm", weather_data.get("rain_24h", weather_data.get("rain", 0))),
                        "tmax": weather_data.get("max_temp_next_24h_c", weather_data.get("temp_max", weather_data.get("temperature", 0))),
                        "wind": weather_data.get("max_wind_next_24h_ms", weather_data.get("wind_speed", weather_data.get("wind", 0)))
                    }
                ))
        
        # NDVI card
        if "ndvi" in result.tool_notes:
            ndvi_data = result.tool_notes["ndvi"]
            if isinstance(ndvi_data, dict):
                cards.append(MobileCard(
                    kind="ndvi",
                    data={
                        "mean": ndvi_data.get("ndvi_latest", ndvi_data.get("mean", 0)),
                        "cov": ndvi_data.get("ndvi_coverage_pct", ndvi_data.get("coverage_pct", 0)),
                        "aoi": ndvi_data.get("aoi_used", ndvi_data.get("aoi_km", 0)),
                        "img": ndvi_data.get("quicklook_url")
                    }
                ))
        
        # NDVI quicklook card
        if "ndvi_quicklook" in result.tool_notes:
            ql_data = result.tool_notes["ndvi_quicklook"]
            if isinstance(ql_data, dict) and ql_data.get("available"):
                # Update existing NDVI card or create new one
                for card in cards:
                    if card.kind == "ndvi":
                        # Convert relative URL to full URL
                        img_url = ql_data.get("url")
                        if img_url and img_url.startswith("/static/"):
                            img_url = f"http://localhost:8000{img_url}"
                        card.data["img"] = img_url
                        break
                else:
                    # No NDVI card exists, create one just for the image
                    img_url = ql_data.get("url")
                    if img_url and img_url.startswith("/static/"):
                        img_url = f"http://localhost:8000{img_url}"
                    cards.append(MobileCard(
                        kind="ndvi",
                        data={
                            "mean": 0,
                            "cov": 0,
                            "aoi": 0,
                            "img": img_url
                        }
                    ))
    
    return MobileAskResponse(
        answer=result.text or "मुझे खुशी होगी आपकी मदद करने में!",
        cards=cards,
        follow_ups=result.followups or [],
        lang=req.lang or "hi"
    )
