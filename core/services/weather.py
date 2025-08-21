# app/services/weather.py
import sys
import os
from typing import Optional, Dict, Any

from backend.app.tools.weather_cached import forecast_24h_cached

class WeatherService:
    """Wrapper around existing weather tools."""

    def __init__(self):
        pass

    async def get_forecast(
        self,
        lat: float,
        lon: float,
        tz: str = "auto",
    ) -> Dict[str, Any]:
        """
        Get next-24h hourly conditions + compact 7-day daily highs/lows.

        Args:
            lat, lon: Coordinates
            tz: Open-Meteo timezone string (e.g. "Asia/Kolkata") or "auto"

        Returns:
            Dict with hourly (next 24h) and daily (7d) blocks, or {"error": "..."}.
        """
        try:
            result = await forecast_24h_cached(lat=lat, lon=lon, tz=tz)
            return result or {}
        except Exception as e:
            print(f"Weather error: {e}")
            return {"error": str(e)}
