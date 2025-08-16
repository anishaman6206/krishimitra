from app.http import get_http_client

async def geocode_text(q: str) -> tuple[float, float] | None:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format": "json", "limit": 1}
    
    # Use the global HTTP client
    client = get_http_client()
    r = await client.get(url, params=params, headers={"User-Agent": "KrishiMitra/1.0"})
    r.raise_for_status()
    arr = r.json()
    
    if not arr: 
        return None
    return float(arr[0]["lat"]), float(arr[0]["lon"])
