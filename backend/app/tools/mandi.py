import os
import json
import asyncio
import datetime as dt
import time
from typing import Optional, Dict, Any, List

import xml.etree.ElementTree as ET
from app.utils.cache import get_json, set_json
from app.http import get_http_client

def t(): return time.perf_counter()

from statistics import median
import datetime as _dt
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from the root of the project
dotenv_path = Path(__file__).parents[3] / '.env'
load_dotenv(dotenv_path)

# -------------------------------
# Configuration
# -------------------------------
DATA_GOV_IN_API_KEY = os.getenv("DATA_GOV_IN_API_KEY")
RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"  # Current Daily Price of Various Commodities

API_BASE = "https://api.data.gov.in/resource"
USER_AGENT = "KrishiMitra/1.0 (+https://krishimitra.example.com)"

# -------------------------------
# Helper Functions
# -------------------------------
def _ensure_api_key():
    """Raises an error if the API key is not configured."""
    if not DATA_GOV_IN_API_KEY:
        raise RuntimeError("DATA_GOV_IN_API_KEY not set in .env file")

def _to_int(x: Any) -> Optional[int]:
    """Safely convert a value to a non-negative integer."""
    try:
        i = int(float(x))
        return i if i >= 0 else None
    except (ValueError, TypeError):
        return None

def _ci_eq(a: Optional[str], b: Optional[str]) -> bool:
    """Case-insensitive string comparison."""
    if not a or not b:
        return False
    return a.strip().lower() == b.strip().lower()

def _parse_ddmmyyyy(s: Optional[str]) -> Optional[dt.date]:
    """Parse a 'dd/mm/yyyy' string into a date object."""
    if not s:
        return None
    try:
        return dt.datetime.strptime(s.strip(), "%d/%m/%Y").date()
    except ValueError:
        return None

def _normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize the structure and types of a record from the API."""
    ad_date = _parse_ddmmyyyy(item.get("arrival_date"))
    return {
        "commodity": item.get("commodity"),
        "state": item.get("state"),
        "district": item.get("district"),
        "market": item.get("market"),
        "arrival_date": ad_date.isoformat() if ad_date else item.get("arrival_date"),
        "variety": item.get("variety"),
        "grade": item.get("grade"),
        "min_price_inr_per_qtl": _to_int(item.get("min_price")),
        "max_price_inr_per_qtl": _to_int(item.get("max_price")),
        "modal_price_inr_per_qtl": _to_int(item.get("modal_price")),
        "unit": "INR/Quintal",
        "source": f"data.gov.in:{RESOURCE_ID}"
    }

# -------------------------------
# Core API Interaction (using XML for reliability)
# -------------------------------
async def _fetch_xml_records(commodity: str, limit: int = 200, offset: int = 0, 
                           state: Optional[str] = None, district: Optional[str] = None, 
                           market: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch a page of XML records for a commodity; return as list of dicts."""
    _ensure_api_key()
    
    # Create cache key for this specific API call
    cache_parts = [commodity, str(limit), str(offset)]
    if state: cache_parts.append(f"state:{state}")
    if district: cache_parts.append(f"district:{district}")
    if market: cache_parts.append(f"market:{market}")
    
    api_cache_key = f"api_call:{'|'.join(cache_parts)}"
    
    # Try cache first (5 minute TTL for API calls)
    from app.utils.cache import get_json, set_json
    cached_result = await get_json(api_cache_key, "price")
    if cached_result:
        print(f"💾 API call cache hit: {len(cached_result)} records")
        return cached_result
    
    url = f"{API_BASE}/{RESOURCE_ID}"
    params = {
        "api-key": DATA_GOV_IN_API_KEY,
        "format": "xml",
        "limit": str(limit),
        "offset": str(offset),
        "filters[commodity]": commodity,
    }
    
    # Add API-level filters to reduce data fetched
    if state:
        params["filters[state]"] = state
    if district:
        params["filters[district]"] = district  
    if market:
        params["filters[market]"] = market
        
    headers = {"User-Agent": USER_AGENT, "Accept": "application/xml"}
    
    # Use the global HTTP client
    client = get_http_client()
    r = await client.get(url, params=params, headers=headers)
    r.raise_for_status()
    root = ET.fromstring(r.text)

    recs_el = root.find("records")
    out: List[Dict[str, Any]] = []
    if recs_el is None:
        return out

    for item in recs_el.findall("item"):
        out.append({
            "state": (item.findtext("state") or "").strip(),
            "district": (item.findtext("district") or "").strip(),
            "market": (item.findtext("market") or "").strip(),
            "commodity": (item.findtext("commodity") or "").strip(),
            "variety": (item.findtext("variety") or "").strip(),
            "grade": (item.findtext("grade") or "").strip(),
            "arrival_date": (item.findtext("arrival_date") or "").strip(),
            "min_price": (item.findtext("min_price") or "").strip(),
            "max_price": (item.findtext("max_price") or "").strip(),
            "modal_price": (item.findtext("modal_price") or "").strip(),
        })
    
    # Cache the API result for 5 minutes to avoid duplicate calls
    await set_json(api_cache_key, out, "price")
    print(f"💾 API call cached: {len(out)} records")
    
    return out

async def _fetch_recent_records(commodity: str, pages: int = 5, page_size: int = 200,
                              state: Optional[str] = None, district: Optional[str] = None,
                              market: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Fetch records using progressive API filtering strategy.
    Try specific filters first, then broaden if insufficient data.
    """
    all_recs: List[Dict[str, Any]] = []
    
    # Progressive filtering strategies (most specific to least specific)
    strategies = [
        {"state": state, "district": district, "market": market, "name": "state+district+market"},
        {"state": state, "district": district, "name": "state+district"},
        {"state": state, "name": "state"},
        {"name": "commodity-only"},
    ]
    
    for strategy in strategies:
        if not strategy.get("state") and strategy["name"] != "commodity-only":
            continue  # Skip if we don't have the required filter data
            
        print(f"🔍 [mandi] Trying strategy: {strategy['name']}")
        t0 = t()
        temp_recs: List[Dict[str, Any]] = []
        recs_total = 0
        
        # Try to fetch with this strategy
        for i in range(pages):
            try:
                page_t = t()
                recs = await _fetch_xml_records(
                    commodity, 
                    limit=page_size, 
                    offset=i * page_size,
                    state=strategy.get("state"),
                    district=strategy.get("district"),
                    market=strategy.get("market")
                )
                page_ms = round((t() - page_t) * 1000)
                print(f"   page {i} {strategy['name']}: {len(recs)} recs in {page_ms}ms")
                
                recs_total += len(recs)
                if not recs:
                    break
                temp_recs.extend(recs)
                
                # If we got decent amount of data, we can stop early
                if recs_total >= 50:  # Reasonable sample size
                    break
                    
            except Exception as e:
                page_ms = round((t() - page_t) * 1000)
                print(f"⚠️  [mandi] Strategy {strategy['name']} page {i} failed in {page_ms}ms: {e}")
                break
        
        strategy_ms = round((t() - t0) * 1000)
        print(f"✓ strategy {strategy['name']} total={recs_total} in {strategy_ms}ms")
        
        if temp_recs:
            print(f"✅ [mandi] Strategy {strategy['name']} returned {len(temp_recs)} records")
            all_recs.extend(temp_recs)
            
            # If we got good data from specific filters, prefer that
            if len(temp_recs) >= 10 and strategy["name"] != "commodity-only":
                break
        else:
            print(f"❌ [mandi] Strategy {strategy['name']} returned no records")
    
    print(f"📊 [mandi] Total records fetched: {len(all_recs)}")
    return all_recs

def _pick_best(
    records: List[Dict[str, Any]],
    state: Optional[str],
    district: Optional[str],
    market: Optional[str],
    variety: Optional[str],
    grade: Optional[str],
    strict_state: bool = True,  # <— new
) -> Optional[Dict[str, Any]]:
    """
    From a list of records, find the best match based on location and other attributes.
    This performs client-side filtering.
    """
    rows = [_normalize_item(r) for r in records if _to_int(r.get("modal_price")) is not None]
    rows.sort(key=lambda r: r.get("arrival_date", ""), reverse=True)

    # restrict to same state if requested
    if strict_state and state:
        rows = [r for r in rows if _ci_eq(r["state"], state)]
        if not rows:
            return None

    # Define tiers of matching, from most specific to least specific
    tiers = [
        lambda r: _ci_eq(r["state"], state) and _ci_eq(r["district"], district) and _ci_eq(r["market"], market) and _ci_eq(r["variety"], variety),
        lambda r: _ci_eq(r["state"], state) and _ci_eq(r["district"], district) and _ci_eq(r["market"], market),
        lambda r: _ci_eq(r["state"], state) and _ci_eq(r["district"], district),
        lambda r: _ci_eq(r["state"], state),
    ]
    
    # only allow nationwide fallback when strict_state is False or no state was provided
    if not strict_state or not state:
        tiers.append(lambda r: True)

    for tier in tiers:
        for row in rows:
            if tier(row):
                return row
    return None

async def latest_price(
    commodity: str,
    district: Optional[str] = None,
    state: Optional[str] = None,
    market: Optional[str] = None,
    variety: Optional[str] = None,
    grade: Optional[str] = None,
    cache_ttl_ok: bool = True,
    debug: bool = False,
    strict_state: bool = True,   # <— new default
) -> Dict[str, Any]:
    """
    Returns the freshest price data by fetching a broad list of records for a commodity
    and then filtering them locally to find the best match.
    """
    start = t()
    
    key = f"mandi:{commodity}:{state}:{district}:{market}:{variety}:{grade}"
    if cache_ttl_ok:
        cached = await get_json(key, "price")
        if cached:
            cache_ms = round((t() - start) * 1000)
            print(f"💾 Price cache hit: {cache_ms}ms")
            return cached

    # Fetch records using progressive API filtering
    fetch_start = t()
    recs = await _fetch_recent_records(commodity, pages=5, page_size=200, 
                                     state=state, district=district, market=market)
    fetch_ms = round((t() - fetch_start) * 1000)
    if debug: print(f"[DEBUG] Fetched {len(recs)} total records for commodity='{commodity}' using progressive filtering.")

    # Find best match
    match_start = t()
    best = _pick_best(recs, state, district, market, variety, grade, strict_state=strict_state)
    match_ms = round((t() - match_start) * 1000)
    
    if not best:
        total_ms = round((t() - start) * 1000)
        print(f"❌ Price lookup failed: {total_ms}ms")
        if state and strict_state:
            raise ValueError(
                f"No recent Agmarknet data for commodity='{commodity}' in state='{state}'. "
                "Try a different nearby market or relax state restriction."
            )
        raise ValueError(
            f"No recent Agmarknet data found for commodity='{commodity}' in the specified area after local filtering."
        )
    
    await set_json(key, best, "price")
    total_ms = round((t() - start) * 1000)
    print(f"⏱️  Price lookup: {total_ms}ms (fetch: {fetch_ms}ms, match: {match_ms}ms)")
    
    return best


# --- add price_series ---

async def price_series(
    commodity: str,
    state: str | None = None,
    district: str | None = None,
    market: str | None = None,
    days: int = 90,
    debug: bool = False
) -> list[dict]:
    """
    Returns a list of {date: 'YYYY-MM-DD', modal_price: int} for the last <= `days` days.
    Aggregates by daily median of matching rows.
    """
    # pull data using progressive API filtering
    recs = await _fetch_recent_records(commodity, pages=5, page_size=200,
                                     state=state, district=district, market=market)
    if debug:
        print(f"[DEBUG] series fetched raw={len(recs)} rows for commodity='{commodity}' using progressive filtering")

    # normalize + (soft) filter by provided geography
    rows = []
    for r in recs:
        nr = _normalize_item(r)
        if state    and not _ci_eq(nr["state"], state):         continue
        if district and not _ci_eq(nr["district"], district):   continue
        if market   and not _ci_eq(nr["market"], market):       continue
        if nr["modal_price_inr_per_qtl"] is None:               continue
        try:
            d = _dt.datetime.strptime(nr["arrival_date"], "%Y-%m-%d").date()
        except Exception:
            continue
        rows.append((d, nr["modal_price_inr_per_qtl"]))

    # if too few after strict filter, relax to state-only, then commodity-only
    if len(rows) < 5 and (district or market):
        if debug: print("[DEBUG] few rows after strict filter → relaxing to state-only")
        return await price_series(commodity, state=state, district=None, market=None, days=days, debug=debug)
    if len(rows) < 5 and state:
        if debug: print("[DEBUG] few rows after state-only → relaxing to commodity-only")
        return await price_series(commodity, state=None, district=None, market=None, days=days, debug=debug)

    by_day = {}
    for d, p in rows:
        by_day.setdefault(d, []).append(p)

    # daily median
    series = []
    cutoff = _dt.date.today() - _dt.timedelta(days=days)
    for d in sorted(by_day.keys()):
        if d < cutoff:
            continue
        series.append({"date": d.isoformat(), "modal_price": int(median(by_day[d]))})

    return series


# -------------------------------
# Command-Line Interface for Testing
# -------------------------------
async def _cli(
    commodity: str,
    district: Optional[str],
    state: Optional[str],
    market: Optional[str],
    variety: Optional[str],
    grade: Optional[str],
    debug: bool
):
    """CLI wrapper to test the latest_price function."""
    # Initialize HTTP client for CLI usage
    from app.http import init_http
    await init_http()
    
    try:
        data = await latest_price(commodity, district, state, market, variety, grade, debug=debug)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch latest commodity prices from Agmarknet via data.gov.in")
    parser.add_argument("--commodity", required=True, help="e.g., 'Tomato'")
    parser.add_argument("--state", default=None, help="e.g., 'Uttar Pradesh'")
    parser.add_argument("--district", default=None, help="e.g., 'Azamgarh'")
    parser.add_argument("--market", default=None, help="e.g., 'Azamgarh'")
    parser.add_argument("--variety", default=None, help="e.g., 'Hybrid'")
    parser.add_argument("--grade", default=None, help="e.g., 'FAQ'")
    parser.add_argument("--debug", action="store_true", help="Enable debug printing")
    
    args = parser.parse_args()
    
    asyncio.run(_cli(
        args.commodity,
        args.district,
        args.state,
        args.market,
        args.variety,
        args.grade,
        args.debug
    ))
