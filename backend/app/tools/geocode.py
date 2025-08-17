import time
from app.http import get_http_client

def t(): return time.perf_counter()

async def geocode_text(q: str) -> tuple[float, float] | None:
    start = t()
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format": "json", "limit": 1}
    
    # Use the global HTTP client
    client = get_http_client()
    r = await client.get(url, params=params, headers={"User-Agent": "KrishiMitra/1.0"})
    r.raise_for_status()
    arr = r.json()
    
    ms = round((t() - start) * 1000)
    result = None if not arr else (float(arr[0]["lat"]), float(arr[0]["lon"]))
    print(f"⏱️  Geocoding '{q}': {ms}ms -> {result}")
    
    return result
