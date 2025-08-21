from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime

# Domain models for your business logic
class PriceInfo(BaseModel):
    commodity: str
    price: Optional[float]
    unit: str = "per_quintal"
    market: Optional[str] = None
    date: Optional[datetime] = None
    grade: Optional[str] = None
    variety: Optional[str] = None

class WeatherInfo(BaseModel):
    location: str
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    precipitation: Optional[float] = None
    forecast_days: int = 1
    conditions: Optional[str] = None

class NDVIInfo(BaseModel):
    location: str
    ndvi_value: Optional[float] = None
    date: Optional[datetime] = None
    aoi_km: float = 1.0
    image_url: Optional[str] = None

class MandiInfo(BaseModel):
    market_name: str
    location: Optional[str] = None
    arrivals: Optional[Dict[str, Any]] = None
    prices: List[PriceInfo] = []
