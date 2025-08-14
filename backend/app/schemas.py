from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Literal

InputType = Literal["text", "voice", "image"]

class ChatRequest(BaseModel):
    type: InputType = Field(..., description="text | voice | image")
    content: Optional[str] = None        # text or base64 for MVP
    media_url: Optional[str] = None
    metadata: Dict[str, Any] = {}

class ChatResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    citations: list[str] = []
    extras: Dict[str, Any] = {}
