from .base import ChannelAdapter
from ..models.io import ChannelMessageIn, ChannelMessageOut

class TelegramAdapter(ChannelAdapter):
    def __init__(self, default_lang="hi"):
        self.default_lang = default_lang

    def to_core(self, update: dict) -> ChannelMessageIn:
        """Convert Telegram update to core message format."""
        m = update.get("message") or {}
        chat = m.get("chat", {})
        text = m.get("text")
        voice = m.get("voice")
        voice_url = None
        
        if voice:
            # You'll fetch file_path via getFile; put temporary placeholder
            voice_url = f"tg://file/{voice['file_id']}"
        
        # Extract location if present
        geo = None
        location = m.get("location")
        if location:
            geo = {
                "lat": location.get("latitude"),
                "lon": location.get("longitude")
            }
        
        return ChannelMessageIn(
            channel="telegram",
            user_id=str(chat.get("id")),
            text=text,
            voice_url=voice_url,
            lang_hint=self.default_lang,
            geo=geo,
            metadata={"raw": update},
        )

    def to_channel(self, out: ChannelMessageOut) -> dict:
        """Convert core output to Telegram-ready payload."""
        payload = {}
        
        if out.text:
            payload["text"] = out.text
            
        # TODO: Add support for attachments as photos/documents
        # TODO: Add support for followups as inline keyboards
        
        # Basic reply markup for followups (simple buttons)
        if out.followups:
            # Convert followups to inline keyboard
            keyboard = []
            for followup in out.followups[:5]:  # Limit to 5 buttons
                keyboard.append([{"text": followup, "callback_data": followup}])
            payload["reply_markup"] = {"inline_keyboard": keyboard}
        
        return payload
