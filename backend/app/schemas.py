from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ---------- Request models ----------

class Geo(BaseModel):
    lat: Optional[float] = Field(None, description="Latitude in WGS84")
    lon: Optional[float] = Field(None, description="Longitude in WGS84")

class AskRequest(BaseModel):
    # User message
    text: str = Field(..., description="User's question or query text")
    lang: Optional[str] = Field(None, description="Language hint (e.g., 'en','hi','kn'). If omitted, auto-detected.")
    geo: Optional[Geo] = Field(None, description="Optional location to enable weather/NDVI")

    # Optional domain hints (help tools be precise)
    crop: Optional[str] = Field(None, description="Commodity/crop name (e.g., 'Tomato')")
    state: Optional[str] = None
    district: Optional[str] = None
    market: Optional[str] = None
    variety: Optional[str] = None
    grade: Optional[str] = None

    # Sell/Wait parameters (optional)
    horizon_days: Optional[int] = Field(None, description="Holding period in days for sell/wait logic")
    qty_qtl: Optional[float] = Field(None, description="Quantity in quintals to estimate value impact")

    # Debug flag
    debug: bool = Field(False, description="If true, tools may include extra debug notes")


# ---------- Response models ----------

class Source(BaseModel):
    title: Optional[str] = None   # human title if available
    source: Optional[str] = None  # filename or URL-ish label
    page: Optional[int] = None    # page number for PDFs
    score: Optional[float] = None # retrieval similarity/relevance score

class AskResponse(BaseModel):
    answer: str
    sources: List[Source] = Field(default_factory=list)           # citations from RAG
    tool_notes: Dict[str, Any] = Field(default_factory=dict)      # raw outputs from tools (price/sell_wait/weather/ndvi/rag)
    follow_ups: List[str] = Field(default_factory=list)
    lang: str = "en"                                              # final answer language code
