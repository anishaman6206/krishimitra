# backend/app/tools/geocode.py
import time
from app.http import get_http_client

def t(): return time.perf_counter()

async def geocode_text(q: str) -> tuple[float, float] | None:
    start = t()
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format": "json", "limit": 1}
    headers = {
        "User-Agent": "KrishiMitra/1.0 (contact: support@krishimitra.example)",
        "Accept-Language": "en-IN"
    }
    client = get_http_client()
    r = await client.get(url, params=params, headers=headers)
    r.raise_for_status()
    arr = r.json()
    result = None if not arr else (float(arr[0]["lat"]), float(arr[0]["lon"]))
    print(f"⏱️  Geocoding '{q}': {round((t()-start)*1000)}ms -> {result}")
    return result


