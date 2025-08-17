import time
import datetime as dt
from zoneinfo import ZoneInfo  # Python 3.9+ built-in timezone support
from typing import Any, Dict, Optional
from app.cache import get_json, set_json
from .mandi import latest_price as _latest_price
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

async def latest_price_cached(
    commodity: str,
    state: Optional[str] = None,
    district: Optional[str] = None,
    market: Optional[str] = None,
    variety: Optional[str] = None,
    grade: Optional[str] = None,
    cache_ttl_ok: bool = True,
    debug: bool = False
) -> Dict[str, Any]:
    start = t()
    k = _price_key(commodity, state, district, market, variety, grade)
    
    hit = await get_json(k, "price")
    if hit:
        cache_ms = round((t() - start) * 1000)
        print(f"💾 Price cache hit: {cache_ms}ms")
        return hit
    
    # Ensure HTTP client is initialized for standalone usage
    try:
        from app.http import get_http_client
        get_http_client()  # Test if client exists
    except RuntimeError:
        from app.http import init_http
        await init_http()
        print("🔗 HTTP client initialized for cached tool")
    
    fresh = await _latest_price(
        commodity, 
        state=state, 
        district=district, 
        market=market, 
        variety=variety, 
        grade=grade,
        cache_ttl_ok=cache_ttl_ok,
        debug=debug
    )
    await set_json(k, fresh, "price")
    
    total_ms = round((t() - start) * 1000)
    print(f"⏱️  Price lookup: {total_ms}ms (fresh)")
    return fresh

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
    
    # Ensure HTTP client is initialized for standalone usage  
    try:
        from app.http import get_http_client
        get_http_client()  # Test if client exists
    except RuntimeError:
        from app.http import init_http
        await init_http()
        print("🔗 HTTP client initialized for cached tool")
    
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