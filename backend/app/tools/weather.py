# backend/app/tools/weather.py
import httpx
from typing import Dict, Any

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

async def forecast_24h(lat: float, lon: float, tz: str = "auto") -> Dict[str, Any]:
    """
    Fetch hourly temp, precip, wind for next 24h from Open-Meteo (no API key).
    Returns a compact summary we can use for irrigation logic.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m",
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,wind_speed_10m_max",
        "forecast_days": 2,   # keeping it light for hackathon
        "timezone": tz,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(OPEN_METEO_URL, params=params)
        r.raise_for_status()
        data = r.json()

    # Build a tiny summary for the next ~24h
    hr = data.get("hourly", {})
    times = hr.get("time", [])[:24]
    temps = hr.get("temperature_2m", [])[:24]
    rains = hr.get("precipitation", [])[:24]
    winds = hr.get("wind_speed_10m", [])[:24]

    total_rain = float(sum(x or 0.0 for x in rains)) if rains else 0.0
    max_temp = float(max(temps)) if temps else None
    max_wind = float(max(winds)) if winds else None

    return {
        "lat": lat, "lon": lon, "times": times,
        "total_rain_next_24h_mm": total_rain,
        "max_temp_next_24h_c": max_temp,
        "max_wind_next_24h_ms": max_wind,
        "source": "Open-Meteo",
    }
