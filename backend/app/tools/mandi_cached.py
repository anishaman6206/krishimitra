# app/tools/mandi_cached.py
import time
import datetime as dt
from zoneinfo import ZoneInfo
from typing import Any, Dict, Optional, List

from app.utils.cache import get_json, set_json
from .mandi import latest_price as _latest_price, price_series as _price_series
from .pricing import advise_sell_or_wait as _sell_or_wait

def t(): return time.perf_counter()

def _today_key(prefix: str) -> str:
    """Generate IST-based date key"""
    ist = ZoneInfo("Asia/Kolkata")
    today_ist = dt.datetime.now(ist).date().isoformat()
    return f"{prefix}:{today_ist}"

def _price_key(commodity, state, district, market, variety, grade) -> str:
    """Generate cache key for price queries"""
    return _today_key(f"mandi:price:{commodity}:{state}:{district}:{market}:{variety}:{grade}")

def _series_key(commodity, state, district, market, days: int) -> str:
    """Cache key for price series"""
    return _today_key(f"mandi:series:{commodity}:{state}:{district}:{market}:d{days}")

async def _ensure_http_client():
    """Ensure HTTP client is initialized for standalone usage."""
    try:
        from app.http import get_http_client
        get_http_client()
    except RuntimeError:
        from app.http import init_http
        await init_http()
        print("🔗 HTTP client initialized for cached tool")

async def latest_price_cached(
    commodity: str,
    state: Optional[str] = None,
    district: Optional[str] = None,
    market: Optional[str] = None,
    variety: Optional[str] = None,
    grade: Optional[str] = None,
    cache_ttl_ok: bool = True,
    strict_state: bool = True,
    debug: bool = False
) -> Dict[str, Any]:
    start = t()
    print(f"💰 latest_price_cached called with:")
    print(f"   commodity: '{commodity}'")
    print(f"   state: '{state}'")
    print(f"   district: '{district}'")
    print(f"   market: '{market}'")
    print(f"   variety: '{variety}'")
    print(f"   grade: '{grade}'")
    
    k = _price_key(commodity, state, district, market, variety, grade)
    print(f"💰 Cache key: {k}")

    hit = await get_json(k, "price")
    if hit:
        cache_ms = round((t() - start) * 1000)
        print(f"💾 Price cache hit: {cache_ms}ms")
        return hit

    await _ensure_http_client()

    fresh = await _latest_price(
        commodity,
        state=state,
        district=district,
        market=market,
        variety=variety,
        grade=grade,
        cache_ttl_ok=cache_ttl_ok,
        strict_state=strict_state,
        debug=debug,
    )
    await set_json(k, fresh, "price")

    total_ms = round((t() - start) * 1000)
    print(f"⏱️  Price lookup: {total_ms}ms (fresh)")
    return fresh

async def price_series_cached(
    commodity: str,
    state: Optional[str] = None,
    district: Optional[str] = None,
    market: Optional[str] = None,
    days: int = 90,
    debug: bool = False
) -> List[Dict[str, Any]]:
    start = t()
    k = _series_key(commodity, state, district, market, days)

    hit = await get_json(k, "price")
    if hit:
        cache_ms = round((t() - start) * 1000)
        print(f"💾 Series cache hit: {cache_ms}ms")
        return hit

    await _ensure_http_client()

    series = await _price_series(
        commodity=commodity,
        state=state,
        district=district,
        market=market,
        days=days,
        debug=debug,
    )
    await set_json(k, series, "price")

    total_ms = round((t() - start) * 1000)
    print(f"⏱️  Series lookup: {total_ms}ms (fresh)")
    return series

async def advise_sell_or_wait_cached(
    commodity: str,
    state: Optional[str] = None,
    district: Optional[str] = None,
    market: Optional[str] = None,
    variety: Optional[str] = None,
    grade: Optional[str] = None,
    horizon_days: int = 7,
    qty_qtl: Optional[float] = None,
    debug: bool = False,
    price_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    start = t()
    k = _price_key(commodity, state, district, market, variety, grade) + f":h{horizon_days}"

    hit = await get_json(k, "price")
    if hit:
        cache_ms = round((t() - start) * 1000)
        print(f"💾 Pricing analysis cache hit: {cache_ms}ms")
        return hit

    await _ensure_http_client()

    fresh = await _sell_or_wait(
        commodity=commodity,
        state=state,
        district=district,
        market=market,
        variety=variety,
        grade=grade,
        horizon_days=horizon_days,
        qty_qtl=qty_qtl,
        debug=debug,
        price_context=price_context
    )
    await set_json(k, fresh, "price")

    total_ms = round((t() - start) * 1000)
    print(f"⏱️  Pricing analysis: {total_ms}ms (fresh)")
    return fresh
