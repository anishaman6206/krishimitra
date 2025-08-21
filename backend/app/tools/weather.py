# backend/app/tools/weather.py
import time
from typing import Dict, Any, List
from app.http import get_http_client

def t(): return time.perf_counter()

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

async def forecast_24h(lat: float, lon: float, tz: str = "auto") -> Dict[str, Any]:
    start = t()
    """
    Hourly next-24h + compact 7-day daily highs/lows (no API key).
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        # hourly for next-24h decisioning
        "hourly": "temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m",
        # daily block for 7-day min/max reasoning
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max",
        "forecast_days": 7,
        "timezone": tz,
        "temperature_unit": "celsius",
        "windspeed_unit": "kmh",   # <-- correct unit param (avoids earlier 500s)
    }
    
    # Use the global HTTP client instead of creating a new one
    api_start = t()
    client = get_http_client()
    r = await client.get(OPEN_METEO_URL, params=params)
    r.raise_for_status()
    api_ms = round((t() - api_start) * 1000)
    
    data = r.json()

    hr = data.get("hourly", {}) or {}
    times = (hr.get("time") or [])[:24]
    temps = (hr.get("temperature_2m") or [])[:24]
    rains = (hr.get("precipitation") or [])[:24]
    winds = (hr.get("wind_speed_10m") or [])[:24]  # km/h due to windspeed_unit

    total_rain = float(sum(x or 0.0 for x in rains)) if rains else 0.0
    max_temp = float(max(temps)) if temps else None
    min_temp = float(min(temps)) if temps else None
    max_wind_kmh = float(max(winds)) if winds else None

    # Build compact 7-day list
    daily = data.get("daily", {}) or {}
    dtime: List[str] = daily.get("time") or []
    tmax:  List[float] = daily.get("temperature_2m_max") or []
    tmin:  List[float] = daily.get("temperature_2m_min") or []
    rain:  List[float] = daily.get("precipitation_sum") or []
    prob:  List[float] = daily.get("precipitation_probability_max") or []
    wmax:  List[float] = daily.get("wind_speed_10m_max") or []

    daily_list = []
    for i in range(min(len(dtime), len(tmax), len(tmin))):
        daily_list.append({
            "date": dtime[i],
            "tmax_c": None if tmax[i] is None else float(tmax[i]),
            "tmin_c": None if tmin[i] is None else float(tmin[i]),
            "rain_mm": None if i >= len(rain) or rain[i] is None else float(rain[i]),
            "rain_chance_pct": None if i >= len(prob) or prob[i] is None else float(prob[i]),
            "wind_kmh_max": None if i >= len(wmax) or wmax[i] is None else float(wmax[i]),
        })

    total_ms = round((t() - start) * 1000)
    print(f"⏱️  Weather forecast: {total_ms}ms (API: {api_ms}ms)")

    return {
        "lat": lat,
        "lon": lon,
        "times": times,
        "total_rain_next_24h_mm": total_rain,
        "max_temp_next_24h_c": max_temp,
        "min_temp_next_24h_c": min_temp,
        "max_wind_next_24h_kmh": max_wind_kmh,
        "max_wind_next_24h_ms": max_wind_kmh / 3.6 if max_wind_kmh else None,  # Convert km/h to m/s
        "daily": daily_list,               # <-- 7-day highs/lows here
        "source": "Open-Meteo",
    }
