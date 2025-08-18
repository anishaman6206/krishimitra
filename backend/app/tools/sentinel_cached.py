# app/tools/sentinel_cached.py
import time
from typing import Optional, Dict, Any, Tuple, Sequence

from app.utils.cache import get_json, set_json, get_bytes, set_bytes
from app.tools.sentinel import sentinel_summary

def t(): return time.perf_counter()

def _round3(x: float) -> str:
    return f"{round(x, 3):.3f}"

def _aoi_sig(steps: Optional[Sequence[int]]) -> str:
    if not steps:
        return "auto"
    return "x".join(str(int(s)) for s in steps)

async def sentinel_summary_cached(
    lat: float,
    lon: float,
    farm_size_meters: int = 200,      # starting AOI edge
    recent_days: int = 45,            # match your sentinel.py default
    resolution: int = 10,
    autogrow: bool = True,
    aoi_steps_m: Tuple[int, ...] = (200, 500, 1000, 2000, 3000),
    min_cov_pct: float = 10.0,
    low_coverage_threshold: int = 50
) -> Dict[str, Any]:
    """
    Cache wrapper for sentinel_summary (NDVI/NDMI/NDWI/LAI means + coverage + advice).
    """
    t0 = t()
    key = (
        f"veg:{_round3(lat)}:{_round3(lon)}:"
        f"aoi{farm_size_meters}:{_aoi_sig(aoi_steps_m)}:"
        f"d{recent_days}:res{resolution}:min{int(min_cov_pct)}"
    )

    cached = await get_json(key, "ndvi")
    if cached is not None:
        print(f"💾 Vegetation cache hit: {round((t()-t0)*1000)}ms")
        return cached

    fresh = await sentinel_summary(
        lat=lat,
        lon=lon,
        farm_size_meters=farm_size_meters,
        recent_days=recent_days,
        resolution=resolution,
        autogrow=autogrow,
        aoi_steps_m=aoi_steps_m,
        min_cov_pct=min_cov_pct,
        low_coverage_threshold=low_coverage_threshold,
    )
    await set_json(key, fresh, "ndvi")
    print(f"⏱️  Vegetation fetch: {round((t()-t0)*1000)}ms (fresh)")
    return fresh

