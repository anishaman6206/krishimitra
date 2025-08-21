from pydantic import BaseModel
from typing import Literal, Optional, Dict, Any, List

Channel = Literal["web", "telegram", "whatsapp"]

class ChannelMessageIn(BaseModel):
    channel: Channel
    user_id: str
    text: Optional[str] = None
    voice_url: Optional[str] = None
    lang_hint: Optional[str] = None
    geo: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = {}

class Attachment(BaseModel):
    kind: Literal["image", "audio"]
    url: str
    caption: Optional[str] = None

class ChannelMessageOut(BaseModel):
    text: Optional[str] = None
    attachments: List[Attachment] = []
    followups: List[str] = []
    tool_notes: Optional[Dict[str, Any]] = None
