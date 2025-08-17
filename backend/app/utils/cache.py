# backend/app/utils/cache.py
import time
import threading
from typing import Any, Optional

class SimpleTTLCache:
    def __init__(self, default_ttl: Optional[int] = 600):
        self._data: dict[str, tuple[Any, Optional[float]]] = {}
        self._default_ttl = default_ttl
        self._lock = threading.Lock()

    def get(self, key: str, default: Any = None) -> Any:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if not item:
                return default
            value, expiry = item
            if expiry is not None and expiry <= now:
                self._data.pop(key, None)
                return default
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            eff_ttl = ttl if ttl is not None else self._default_ttl
            expiry = (time.time() + eff_ttl) if eff_ttl is not None else None
            self._data[key] = (value, expiry)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

cache = SimpleTTLCache(default_ttl=600)


# --- Flush helpers ---

def flush_all() -> int:
    """
    Clear the entire cache. Returns number of keys flushed.
    Works for mapping-like caches (dict, TTLCache, etc.).
    """
    try:
        n = len(cache)
    except Exception:
        n = 0
    # most caches expose clear()
    if hasattr(cache, "clear"):
        cache.clear()
    else:
        # fallback: delete keys one by one
        try:
            for k in list(cache.keys()):
                del cache[k]
        except Exception:
            pass
    return n

def flush_prefix(prefix: str) -> int:
    """
    Remove only keys that start with `prefix` (e.g., 'mandi:', 'weather:', 'ndvi:').
    Returns number of keys removed.
    """
    removed = 0
    try:
        keys = list(cache.keys())
    except Exception:
        return 0
    for k in keys:
        if str(k).startswith(prefix):
            try:
                del cache[k]
                removed += 1
            except Exception:
                pass
    return removed
