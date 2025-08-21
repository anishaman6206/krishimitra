"""
Core caching utilities - abstraction over the existing cache.
"""
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../backend'))

try:
    from app.utils.cache import cache_get, cache_set, cache_exists
except ImportError:
    # Fallback in-memory cache
    _cache = {}
    
    def cache_get(key):
        return _cache.get(key)
    
    def cache_set(key, value, ttl=3600):
        _cache[key] = value
    
    def cache_exists(key):
        return key in _cache

class CacheService:
    """Simple cache abstraction."""
    
    @staticmethod
    def get(key: str):
        """Get cached value."""
        return cache_get(key)
    
    @staticmethod 
    def set(key: str, value, ttl: int = 3600):
        """Set cached value with TTL."""
        return cache_set(key, value, ttl)
    
    @staticmethod
    def exists(key: str) -> bool:
        """Check if key exists in cache."""
        return cache_exists(key)
    
    @staticmethod
    def delete(key: str):
        """Delete cached value."""
        global _cache
        if key in _cache:
            del _cache[key]
