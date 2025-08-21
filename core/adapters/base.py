from typing import Protocol
from ..models.io import ChannelMessageIn, ChannelMessageOut

class ChannelAdapter(Protocol):
    def to_core(self, raw_update: dict) -> ChannelMessageIn: 
        """Convert channel-specific input to core model."""
        ...
    
    def to_channel(self, out: ChannelMessageOut) -> dict:
        """Return dict ready to send via channel API (text + options)."""
        ...
