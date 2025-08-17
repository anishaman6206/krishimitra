# app/tools/weather_cached.py
import datetime as dt
from time import perf_counter
from typing import Dict, Any
from app.utils.cache import get_json, set_json
from .weather import forecast_24h as _wx

def t(): return perf_counter()

def _daykey(lat: float, lon: float, tz: str) -> str:
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    # round to ~8–12km cell to increase hit-rate
    latr = round(lat, 2)
    lonr = round(lon, 2)
    return f"wx:{latr}:{lonr}:{today}:{tz}"

async def forecast_24h_cached(lat: float, lon: float, tz: str = "auto") -> Dict[str, Any]:
    start = t()
    key = _daykey(lat, lon, tz)
    
    hit = await get_json(key, "weather")
    if hit:
        cache_ms = round((t() - start) * 1000)
        print(f"⏱️  Weather forecast: {cache_ms}ms (cached)")
        return hit
    
    # Ensure HTTP client is initialized for standalone usage
    try:
        from app.http import get_http_client
        get_http_client()  # Test if client exists
    except RuntimeError:
        from app.http import init_http
        await init_http()
        print("🔗 HTTP client initialized for cached tool")
    
    fresh = await _wx(lat, lon, tz=tz)
    # Use weather cache type for automatic 6h TTL
    await set_json(key, fresh, "weather")
    
    total_ms = round((t() - start) * 1000)
    print(f"⏱️  Weather forecast: {total_ms}ms (fresh)")
    return fresh
