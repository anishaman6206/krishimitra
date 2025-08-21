# app/tools/sentinel_cached.py
import time
from typing import Optional, Dict, Any
from app.utils.cache import get_json, set_json, get_bytes, set_bytes
from app.tools.sentinel import ndvi_snapshot, ndvi_quicklook, sentinel_summary

def t(): return time.perf_counter()

async def ndvi_snapshot_cached(
    lat: float, 
    lon: float, 
    aoi_km: float = 0.5, 
    recent_days: int = 14, 
    prev_days: int = 14, 
    gap_days: int = 30
) -> Dict[str, Any]:
    """
    Cached wrapper for NDVI snapshot analysis.
    Cache TTL: 7 days (satellite data doesn't change frequently)
    """
    t0 = t()
    
    # Build cache key with rounded coordinates for better hit rate
    lat_rounded = round(lat, 3)  # ~100m precision
    lon_rounded = round(lon, 3)
    cache_key = f"ndvi_snapshot:{lat_rounded}:{lon_rounded}:aoi{aoi_km}:r{recent_days}:p{prev_days}:g{gap_days}"
    
    # Try cache first
    cached = await get_json(cache_key, "ndvi")
    if cached is not None:
        cache_ms = round((t() - t0) * 1000)
        print(f"💾 NDVI snapshot cache hit: {cache_ms}ms")
        return cached
    
    # Cache miss - fetch fresh data
    print(f"🛰️  Fetching fresh NDVI snapshot for lat={lat}, lon={lon}, aoi={aoi_km}km...")
    fresh = await ndvi_snapshot(
        lat=lat,
        lon=lon,
        aoi_km=aoi_km,
        recent_days=recent_days,
        prev_days=prev_days,
        gap_days=gap_days
    )
    
    # Cache the result
    await set_json(cache_key, fresh, "ndvi")
    
    total_ms = round((t() - t0) * 1000)
    aoi_used = fresh.get("aoi_used", aoi_km)
    print(f"⏱️  NDVI snapshot: {total_ms}ms (fresh, AOI: {aoi_used}km)")
    
    return fresh

async def ndvi_quicklook_cached(
    lat: float, 
    lon: float, 
    aoi_km: Optional[float] = None, 
    recent_days: int = 7
) -> Optional[bytes]:
    """
    Cached wrapper for NDVI quicklook image generation.
    Cache TTL: 7 days (satellite images don't change frequently)
    """
    t0 = t()
    
    # Build cache key with rounded coordinates for better hit rate
    lat_rounded = round(lat, 3)  # ~100m precision
    lon_rounded = round(lon, 3)
    aoi_str = f"{aoi_km}" if aoi_km else "auto"
    cache_key = f"ndvi_quicklook:{lat_rounded}:{lon_rounded}:aoi{aoi_str}:r{recent_days}"
    
    # Try cache first
    cached = await get_bytes(cache_key, "ndvi")
    if cached is not None:
        cache_ms = round((t() - t0) * 1000)
        print(f"💾 NDVI quicklook cache hit: {cache_ms}ms")
        return cached
    
    # Cache miss - fetch fresh data
    print(f"🛰️  Generating fresh NDVI quicklook for lat={lat}, lon={lon}...")
    
    # Use default if aoi_km is None
    actual_aoi_km = aoi_km if aoi_km is not None else 0.5
    fresh = await ndvi_quicklook(
        lat=lat,
        lon=lon,
        aoi_km=actual_aoi_km,
        recent_days=recent_days
    )
    
    if fresh is not None:
        # Cache the result
        await set_bytes(cache_key, fresh, "ndvi")
        
        total_ms = round((t() - t0) * 1000)
        size_mb = len(fresh) / (1024 * 1024)
        print(f"⏱️  NDVI quicklook: {total_ms}ms (fresh, {size_mb:.1f}MB)")
    else:
        total_ms = round((t() - t0) * 1000)
        print(f"⏱️  NDVI quicklook: {total_ms}ms (no data available)")
    
    return fresh


async def sentinel_summary_cached(
    lat: float,
    lon: float,
    farm_size_meters: int = 200,
    recent_days: int = 20,
    resolution: int = 10,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Cached wrapper for comprehensive sentinel satellite summary.
    Includes NDVI, NDMI, NDWI, LAI with farmer-friendly advice.
    Cache TTL: 7 days (168 hours)
    """
    t0 = t()
    
    # Build cache key with rounded coordinates
    lat_rounded = round(lat, 3)  # ~100m precision
    lon_rounded = round(lon, 3)
    cache_key = f"sentinel_summary:{lat_rounded}:{lon_rounded}:size{farm_size_meters}:days{recent_days}:res{resolution}"
    
    # Try cache first
    cached = await get_json(cache_key, "sentinel")
    if cached is not None:
        total_ms = round((t() - t0) * 1000)
        print(f"💾 Sentinel summary cached for 168h: {cache_key[:60]}...")
        print(f"⏱️  Sentinel summary: {total_ms}ms (cached)")
        return cached
    
    # Cache miss - fetch fresh comprehensive data
    print(f"🛰️  Fetching fresh comprehensive sentinel data for lat={lat}, lon={lon}...")
    
    try:
        fresh = await sentinel_summary(
            lat=lat,
            lon=lon,
            farm_size_meters=farm_size_meters,
            recent_days=recent_days,
            resolution=resolution
        )
        
        if fresh:
            # Cache for 7 days (check what parameters set_json actually takes)
            await set_json(cache_key, fresh, "sentinel")
            
            total_ms = round((t() - t0) * 1000)
            print(f"⏱️  Sentinel summary: {total_ms}ms (fresh)")
            
            if debug:
                fresh["_debug"] = {
                    "cache_key": cache_key,
                    "fetch_time_ms": total_ms,
                    "farm_size_m": farm_size_meters,
                    "days_window": recent_days
                }
            
            return fresh
        else:
            total_ms = round((t() - t0) * 1000)
            print(f"⏱️  Sentinel summary: {total_ms}ms (no data)")
            return {"error": "No satellite data available"}
            
    except Exception as e:
        total_ms = round((t() - t0) * 1000)
        print(f"❌ Sentinel summary error: {e}")
        print(f"⏱️  Sentinel summary: {total_ms}ms (error)")
        return {"error": str(e)}
