# backend/app/utils/cache.py
import time
import threading
import asyncio
import datetime as dt
from typing import Any, Optional, Iterable

# -----------------------------
# Core simple in-memory TTL cache (sync)
# -----------------------------
class SimpleTTLCache:
    def __init__(self, default_ttl: Optional[int] = 600):
        # _data: key -> (value, expiry_ts or None)
        self._data: dict[str, tuple[Any, Optional[float]]] = {}
        self._default_ttl = default_ttl
        self._lock = threading.Lock()

    def _now(self) -> float:
        return time.time()

    def get(self, key: str, default: Any = None) -> Any:
        now = self._now()
        with self._lock:
            item = self._data.get(key)
            if not item:
                return default
            value, expiry = item
            if expiry is not None and expiry <= now:
                # expired → drop
                self._data.pop(key, None)
                return default
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value with optional per-key TTL."""
        with self._lock:
            eff_ttl = ttl if ttl is not None else self._default_ttl
            expiry = (self._now() + eff_ttl) if eff_ttl is not None else None
            self._data[key] = (value, expiry)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    # Helpers to support flush utilities
    def keys(self) -> Iterable[str]:
        with self._lock:
            return list(self._data.keys())

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def sweep(self) -> int:
        """Remove expired keys proactively; returns count removed."""
        now = self._now()
        removed = 0
        with self._lock:
            for k in list(self._data.keys()):
                _, expiry = self._data[k]
                if expiry is not None and expiry <= now:
                    self._data.pop(k, None)
                    removed += 1
        return removed


# Singleton cache instance used everywhere
cache = SimpleTTLCache(default_ttl=600)


# -----------------------------
# Async convenience layer with type-specific TTLs (feature branch)
# -----------------------------

# In this prototype we do not use Redis; only in-memory.
USE_REDIS = False
_redis = None  # placeholder for future

# Agricultural data cache TTLs (seconds)
CACHE_TTL = {
    "price":   24 * 3600,   # daily
    "weather": 6  * 3600,   # ~4x/day updates
    "ndvi":    7  * 24 * 3600,  # weekly-ish
    "default": 3600,
}

def _now_ts() -> float:
    return dt.datetime.utcnow().timestamp()

def _get_ttl(cache_type: str) -> int:
    return int(CACHE_TTL.get(cache_type, CACHE_TTL["default"]))

async def init_cache():
    """Init hook (kept async for symmetry with future Redis mode)."""
    print("💾 Cache initialized (in-memory mode)")
    # Optionally run a lightweight periodic sweeper
    asyncio.create_task(_cleanup_task())

async def _cleanup_task():
    while True:
        try:
            await asyncio.sleep(3600)
            n = cache.sweep()
            if n:
                print(f"🧹 Cache sweep removed {n} expired keys")
        except Exception as e:
            print(f"Cache sweep error: {e}")

# --- JSON helpers ---

async def get_json(key: str, cache_type: str = "default") -> Optional[Any]:
    try:
        val = cache.get(key)
        if val is not None:
            # simple log for visibility
            print(f"💾 {cache_type.title()} cache hit: {key}")
        return val
    except Exception as e:
        print(f"Cache get_json error for {key}: {e}")
        return None

async def set_json(key: str, val: Any, cache_type: str = "default"):
    try:
        ttl_sec = _get_ttl(cache_type)
        cache.set(key, val, ttl=ttl_sec)
        hrs = int(ttl_sec / 3600)
        print(f"💾 {cache_type.title()} cached for {hrs}h: {key}")
    except Exception as e:
        print(f"Cache set_json error for {key}: {e}")

# --- Bytes helpers (e.g., NDVI images) ---

async def get_bytes(key: str, cache_type: str = "default") -> Optional[bytes]:
    try:
        val = cache.get(key)
        if isinstance(val, (bytes, bytearray)):
            print(f"💾 {cache_type.title()} image cache hit: {key}")
            return bytes(val)
        return None
    except Exception as e:
        print(f"Cache get_bytes error for {key}: {e}")
        return None

async def set_bytes(key: str, val: bytes, cache_type: str = "default"):
    try:
        ttl_sec = _get_ttl(cache_type)
        cache.set(key, bytes(val), ttl=ttl_sec)
        size_mb = len(val) / (1024 * 1024)
        hrs = int(ttl_sec / 3600)
        print(f"💾 {cache_type.title()} image cached ({size_mb:.2f} MB) for {hrs}h: {key}")
    except Exception as e:
        print(f"Cache set_bytes error for {key}: {e}")


# -----------------------------
# Flush utilities (kept from main and generalized)
# -----------------------------

def flush_all() -> int:
    """Clear entire cache; returns count of keys flushed."""
    try:
        n = len(cache)
    except Exception:
        n = 0
    cache.clear()
    return n

def flush_prefix(prefix: str) -> int:
    """Remove only keys starting with `prefix` (e.g. 'mandi:', 'weather:', 'ndvi:')."""
    removed = 0
    try:
        for k in list(cache.keys()):
            if str(k).startswith(prefix):
                cache.delete(k)
                removed += 1
    except Exception:
        pass
    return removed
