"""
Dependency injection container for the application.
Constructs singletons and provides them to routes/handlers.
"""

import sys
import os

# Add paths for imports - handle both from backend/ and from repo root
current_dir = os.path.dirname(__file__)
backend_root = os.path.abspath(os.path.join(current_dir, '..'))
repo_root = os.path.abspath(os.path.join(current_dir, '../..'))

for path in [backend_root, repo_root]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from core.adapters.telegram import TelegramAdapter
    from core.services.conversation import ConversationService
    from core.services.rag import RAGService
    from core.services.pricing import PricingService
    from core.services.weather import WeatherService
    from core.services.ndvi import NDVIService
    from core.services.mandi import MandiService
except ImportError as e:
    print(f"Warning: Could not import core services: {e}")
    # Create dummy classes for fallback
    class TelegramAdapter:
        def __init__(self, default_lang="hi"):
            pass
    class ConversationService:
        def __init__(self, *args, **kwargs):
            pass
        async def handle(self, msg):
            return {"text": "Service unavailable", "tool_notes": {}}
    RAGService = PricingService = WeatherService = NDVIService = MandiService = ConversationService

# Import existing HTTP client
try:
    from app.http import get_http_client
except ImportError:
    def get_http_client():
        try:
            import httpx
            return httpx.AsyncClient()
        except ImportError:
            return None

# Singletons - created once and reused
_telegram_adapter = None
_rag_service = None
_pricing_service = None
_weather_service = None
_ndvi_service = None
_mandi_service = None
_conversation_service = None

def get_telegram_adapter() -> TelegramAdapter:
    """Get singleton Telegram adapter."""
    global _telegram_adapter
    if _telegram_adapter is None:
        _telegram_adapter = TelegramAdapter(default_lang="hi")
    return _telegram_adapter

def get_rag_service() -> RAGService:
    """Get singleton RAG service."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

def get_pricing_service() -> PricingService:
    """Get singleton pricing service."""
    global _pricing_service
    if _pricing_service is None:
        _pricing_service = PricingService()
    return _pricing_service

def get_weather_service() -> WeatherService:
    """Get singleton weather service."""
    global _weather_service
    if _weather_service is None:
        _weather_service = WeatherService()
    return _weather_service

def get_ndvi_service() -> NDVIService:
    """Get singleton NDVI service."""
    global _ndvi_service
    if _ndvi_service is None:
        _ndvi_service = NDVIService()
    return _ndvi_service

def get_mandi_service() -> MandiService:
    """Get singleton mandi service."""
    global _mandi_service
    if _mandi_service is None:
        _mandi_service = MandiService()
    return _mandi_service

def get_conversation_service() -> ConversationService:
    """Get singleton conversation service."""
    global _conversation_service
    if _conversation_service is None:
        _conversation_service = ConversationService(
            rag=get_rag_service(),
            pricing=get_pricing_service(),
            weather=get_weather_service(),
            ndvi=get_ndvi_service(),
            mandi=get_mandi_service()
        )
    return _conversation_service

def get_http():
    """Get HTTP client (existing function)."""
    return get_http_client()
