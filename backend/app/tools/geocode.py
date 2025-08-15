import httpx

async def geocode_text(q: str) -> tuple[float, float] | None:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format": "json", "limit": 1}
    async with httpx.AsyncClient(timeout=15, headers={"User-Agent": "KrishiMitra/1.0"}) as c:
        r = await c.get(url, params=params)
        r.raise_for_status()
        arr = r.json()
    if not arr: return None
    return float(arr[0]["lat"]), float(arr[0]["lon"])
