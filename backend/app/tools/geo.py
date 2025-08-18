# backend/app/tools/geo.py
import asyncio, time
from typing import Optional, Dict, Any
from app.http import get_http_client
from app.utils.cache import get_json, set_json

USER_AGENT = "KrishiMitra/1.0 (reverse-geocode)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"

# Be a good citizen: Nominatim asks for 1 req/sec per IP; we also cache.
async def reverse_geocode_admin(lat: float, lon: float) -> Dict[str, Optional[str]]:
    """
    Returns {'state': str|None, 'district': str|None, 'city': str|None} from lat/lon.
    Uses Nominatim + in-memory cache.
    """
    key = f"geo:rev:{round(lat,4)}:{round(lon,4)}"
    cached = await get_json(key, cache_type="default")
    if cached:
        return cached

    # polite delay to avoid hammering if called in bursts
    await asyncio.sleep(1.0)

    client = get_http_client()
    params = {
        "lat": str(lat),
        "lon": str(lon),
        "format": "json",
        "zoom": "12",
        "addressdetails": "1",
    }
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    r = await client.get(NOMINATIM_URL, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json() or {}
    addr = data.get("address") or {}

    # Districts come under different keys depending on region
    district = (
        addr.get("state_district")
        or addr.get("district")
        or addr.get("county")
        or addr.get("region")
    )
    state = addr.get("state")
    city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("hamlet")

    out = {"state": state, "district": district, "city": city}
    await set_json(key, out, cache_type="default")
    return out
