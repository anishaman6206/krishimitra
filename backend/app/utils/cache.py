import os, json, asyncio, datetime as dt
from typing import Any, Optional
from contextlib import asynccontextmanager

# Disable Redis for prototype - use in-memory only
USE_REDIS = False  # bool(os.getenv("REDIS_URL"))
_redis = None
_mem: dict[str, tuple[float, Any]] = {}  # expires_at, value

# Agricultural data cache TTLs (in seconds)
CACHE_TTL = {
    "price": 86400,      # 24 hours - prices don't change frequently
    "weather": 21600,    # 6 hours - weather forecasts update 4x daily
    "ndvi": 604800,      # 7 days - satellite data is weekly
    "default": 3600      # 1 hour - general cache
}

async def init_cache():
    """Initialize cache - simplified for in-memory only"""
    global _redis
    print("💾 Cache initialized (in-memory mode for prototype)")
    # Start background cleanup task
    asyncio.create_task(_cleanup_expired_cache())

def _now_ts() -> float: 
    return dt.datetime.utcnow().timestamp()

def _get_ttl(cache_type: str) -> int:
    """Get appropriate TTL based on data type"""
    return CACHE_TTL.get(cache_type, CACHE_TTL["default"])

async def get_json(key: str, cache_type: str = "default") -> Optional[Any]:
    try:
        # Use in-memory cache only
        hit = _mem.get(key)
        if not hit: 
            return None
        expires_at, val = hit
        if _now_ts() > expires_at:
            _mem.pop(key, None)
            return None
        
        # Log cache hit with type info
        time_left = expires_at - _now_ts()
        hours_left = int(time_left / 3600)
        print(f"💾 {cache_type.title()} cache hit: 0ms (expires in {hours_left}h)")
        return val
    except Exception as e:
        print(f"Cache get error for {key}: {e}")
        return None

async def set_json(key: str, val: Any, cache_type: str = "default"):
    """Set cached JSON data with type-specific TTL"""
    try:
        ttl_sec = _get_ttl(cache_type)
        expires_at = _now_ts() + ttl_sec
        _mem[key] = (expires_at, val)
        
        hours = int(ttl_sec / 3600)
        print(f"💾 {cache_type.title()} cached for {hours}h: {key[:50]}...")
    except Exception as e:
        print(f"Cache set error for {key}: {e}")

async def get_bytes(key: str, cache_type: str = "default") -> Optional[bytes]:
    """Get cached binary data (for NDVI images)"""
    try:
        # Use in-memory cache only
        hit = _mem.get(key)
        if not hit: 
            return None
        exp, val = hit
        if _now_ts() > exp: 
            _mem.pop(key, None)
            return None
        print(f"💾 {cache_type.title()} image cache hit: 0ms")
        return val
    except Exception as e:
        print(f"Cache get_bytes error for {key}: {e}")
        return None

async def set_bytes(key: str, val: bytes, cache_type: str = "default"):
    """Set cached binary data with type-specific TTL"""
    try:
        ttl_sec = _get_ttl(cache_type)
        _mem[key] = (_now_ts() + ttl_sec, val)
        
        size_mb = len(val) / (1024 * 1024)
        hours = int(ttl_sec / 3600)
        print(f"💾 {cache_type.title()} image cached ({size_mb:.1f}MB) for {hours}h")
    except Exception as e:
        print(f"Cache set_bytes error for {key}: {e}")

async def _cleanup_expired_cache():
    """Background task to clean up expired cache entries"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            
            now = _now_ts()
            expired_keys = [
                key for key, (expires_at, _) in _mem.items() 
                if now > expires_at
            ]
            
            for key in expired_keys:
                _mem.pop(key, None)
            
            if expired_keys:
                print(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            print(f"Cache cleanup error: {e}")
