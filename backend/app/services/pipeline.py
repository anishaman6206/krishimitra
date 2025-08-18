import asyncio
import time
from typing import List, Dict, Any, Optional

from app.schemas import AskRequest, AskResponse, Source
from app.tools.lang import detect_lang, translate_to_en, translate_from_en
from app.rag.retrieve import retrieve
from app.rag.generate import synthesize

#  Use cached wrappers
from app.tools.mandi_cached import (
    latest_price_cached as latest_price,
    advise_sell_or_wait_cached as advise_sell_or_wait,
)
from app.tools.sentinel_cached import sentinel_summary_cached as sentinel_summary
from app.tools.weather_cached import forecast_24h_cached as wx_forecast


#from app.tools.sentinel import sentinel_summary
from app.config import settings

# ------------ timing helper ------------
def t() -> float:
    return time.perf_counter()

# ------------ fan-out helpers ------------

async def _run_rag(query_en: str, k: int = 4):
    t0 = t()
    # retrieve() is sync; run in a thread for true parallelism
    result = await asyncio.to_thread(lambda: retrieve(query_en, k=k))
    print(f"⏱️  RAG retrieve: {round((t() - t0) * 1000)}ms")
    return result

async def _run_price(req: AskRequest):
    t0 = t()
    result = await latest_price(
        req.crop or settings.DEFAULT_CROP,
        district=req.district,
        state=req.state,
        market=req.market,
        variety=req.variety,
        grade=req.grade,
        debug=req.debug,
    )
    print(f"⏱️  Price lookup: {round((t() - t0) * 1000)}ms")
    return result

async def _run_sell_wait(req: AskRequest, price_ctx: Optional[Dict[str, Any]] = None, horizon_days: Optional[int] = None):
    t0 = t()
    result = await advise_sell_or_wait(
        commodity=req.crop or settings.DEFAULT_CROP,
        state=req.state, district=req.district, market=req.market,
        variety=req.variety, grade=req.grade,
        horizon_days=horizon_days or (req.horizon_days or settings.SELLWAIT_DEFAULT_HORIZON_DAYS),
        qty_qtl=req.qty_qtl,
        debug=req.debug,
        price_context=price_ctx
    )
    print(f"⏱️  Pricing analysis: {round((t() - t0) * 1000)}ms")
    return result

async def _run_weather(req: AskRequest):
    t0 = t()
    g = req.geo or {}
    if not g or g.lat is None or g.lon is None:
        raise ValueError("weather: missing lat/lon")
    result = await wx_forecast(g.lat, g.lon, tz="auto")
    print(f"⏱️  Weather forecast: {round((t() - t0) * 1000)}ms")
    return result

async def _run_veg(req: AskRequest):
    t0 = t()
    g = req.geo or {}
    if not g or g.lat is None or g.lon is None:
        raise ValueError("veg: missing lat/lon")

    start_km = getattr(settings, "SENTINEL_START_AOI_KM", 0.5)
    max_km   = getattr(settings, "SENTINEL_MAX_AOI_KM", 3.0)
    start_m  = int(start_km * 1000)
    max_m    = int(max_km * 1000)

    canonical = [200, 500, 1000, 2000, 3000]
    steps = [start_m] + [s for s in canonical if s >= start_m and s <= max_m]
    steps = tuple(dict.fromkeys(steps))

    # cached call
    return await sentinel_summary(
        lat=g.lat,
        lon=g.lon,
        farm_size_meters=start_m,
        recent_days=45,
        resolution=10,                                             
        autogrow=True,
        aoi_steps_m=steps,
        min_cov_pct=5.0,
    )

    print(f"⏱️  NDVI/NDMI/NDWI/LAI analysis: {round((t() - t0) * 1000)}ms")
    return result

# ------------ summarization helpers ------------

def _fmt_weather(wx: Dict[str, Any], days_head: int = 3) -> str:
    if not isinstance(wx, dict):
        return "WEATHER: unavailable"

    r = float(wx.get("total_rain_next_24h_mm", 0) or 0.0)
    tmax = wx.get("max_temp_next_24h_c")
    if "max_wind_next_24h_kmh" in wx and isinstance(wx["max_wind_next_24h_kmh"], (int, float)):
        w_kmh = float(wx["max_wind_next_24h_kmh"])
    elif "max_wind_next_24h_ms" in wx and isinstance(wx["max_wind_next_24h_ms"], (int, float)):
        w_kmh = float(wx["max_wind_next_24h_ms"]) * 3.6
    else:
        w_kmh = None
    wtxt = f"{w_kmh:.0f} km/h" if isinstance(w_kmh, float) else "—"

    line = f"WEATHER 24h: rain≈{r:.0f} mm, max temp {tmax if tmax is not None else '—'}°C, max wind {wtxt}."

    daily = wx.get("daily") or []
    if isinstance(daily, list) and daily:
        try:
            tmins = [(d.get("tmin_c"), d.get("date")) for d in daily if d.get("tmin_c") is not None]
            tmaxs = [(d.get("tmax_c"), d.get("date")) for d in daily if d.get("tmax_c") is not None]
            if tmins and tmaxs:
                min_t, min_d = min(tmins, key=lambda x: x[0])
                max_t, max_d = max(tmaxs, key=lambda x: x[0])
                line += f" NEXT 7d: coldest ~{min_t:.0f}°C ({min_d}), hottest ~{max_t:.0f}°C ({max_d})."
        except Exception:
            pass
    return line

def _fmt_veg(veg: dict) -> str:
    adv = (veg or {}).get("advice") or {}
    lines = adv.get("summary_lines") or []
    cov = (veg or {}).get("coverage_pct") or {}
    aoi_m = (veg or {}).get("aoi_m")
    bits = []
    if lines: bits.append("; ".join(lines[:3]))
    if isinstance(cov, dict) and "ndvi" in cov: bits.append(f"NDVI cov {cov['ndvi']}%")
    if aoi_m: bits.append(f"AOI {aoi_m}m")
    return "FIELD: " + ("; ".join(bits) if bits else "data unavailable")

def _summarize_tools(results: Dict[str, Any], horizon_days: int) -> str:
    lines: List[str] = []

    # price
    p = results.get("price")
    if isinstance(p, dict) and p.get("modal_price_inr_per_qtl") is not None:
        lines.append(
            "PRICE: {commodity} modal ₹{modal}/qtl (min {minp}, max {maxp}) at {mkt}, {dist}, {st} on {dt}."
            .format(
                commodity=p.get("commodity"),
                modal=p.get("modal_price_inr_per_qtl"),
                minp=p.get("min_price_inr_per_qtl"),
                maxp=p.get("max_price_inr_per_qtl"),
                mkt=p.get("market"), dist=p.get("district"),
                st=p.get("state"), dt=p.get("arrival_date"),
            )
        )
    elif isinstance(p, dict) and p.get("error"):
        lines.append(f"PRICE: error={p['error']}")

    def _fmt_inr(x):
        return "—" if x is None else f"₹{int(round(float(x)))}"

    def _trend_word(slope: float | None) -> str:
        if slope is None: return "trend: n/a"
        if slope > 1:     return "trend: rising"
        if slope < -1:    return "trend: falling"
        return "trend: flat"

    # Use last day in forecast triplet (expected, low, high)
    def _last_forecast_triplet(sw: dict):
        fc = sw.get("forecast") or []
        if isinstance(fc, list) and fc:
            last = fc[-1]
            exp  = last.get("p50_adj", last.get("p50"))
            low  = last.get("p20_adj", last.get("p20"))
            high = last.get("p80_adj", last.get("p80"))
            return exp, low, high
        return sw.get("expected_p50_h"), sw.get("band_p20_h"), sw.get("band_p80_h")

    # field (sentinel)
    veg = results.get("veg")
    if veg is not None:
        lines.append(_fmt_veg(veg))

    # SELL/WAIT – plain language (no p50/p80 terms)
    sw = results.get("sell_wait")
    if isinstance(sw, dict) and sw.get("decision"):
        H = horizon_days or settings.SELLWAIT_DEFAULT_HORIZON_DAYS
        span = "WEEK 1" if H == 7 else ("WEEK 2" if H == 14 else f"{H}-DAY")
        notes = sw.get("notes") or {}
        exp, low, high = _last_forecast_triplet(sw)
        line = (
            f"{span}: {'WAIT' if sw['decision']=='WAIT' else 'SELL NOW'} | "
            f"today {_fmt_inr(sw.get('now_price'))}, expected {_fmt_inr(exp)}, "
            f"{_trend_word(notes.get('trend_slope_inr_per_day'))}."
        )
        lines.append(line)
        if isinstance(sw.get("chart"), dict) and sw["chart"].get("url"):
            lines.append(f"FORECAST_IMG: {sw['chart']['url']}")
    elif isinstance(sw, dict) and sw.get("error"):
        lines.append(f"SELL/WAIT: error={sw['error']}")

    if settings.SELLWAIT_INCLUDE_2WEEK:
        sw14 = results.get("sell_wait_14")
        if isinstance(sw14, dict) and sw14.get("decision"):
            notes14 = sw14.get("notes") or {}
            exp14, low14, high14 = _last_forecast_triplet(sw14)
            line14 = (
                f"WEEK 2: {'WAIT' if sw14['decision']=='WAIT' else 'SELL NOW'} | "
                f"today {_fmt_inr(sw14.get('now_price'))}, expected {_fmt_inr(exp14)}, "
                f"{_trend_word(notes14.get('trend_slope_inr_per_day'))}."
            )
            lines.append(line14)
            if isinstance(sw14.get("chart"), dict) and sw14["chart"].get("url"):
                lines.append(f"FORECAST_IMG_14D: {sw14['chart']['url']}")
        elif isinstance(sw14, dict) and sw14.get("error"):
            lines.append(f"SELL/WAIT_14D: error={sw14['error']}")

    # weather
    wx = results.get("weather")
    if isinstance(wx, dict) and ("total_rain_next_24h_mm" in wx or "daily" in wx):
        lines.append(_fmt_weather(wx, days_head=getattr(settings, "WX_SUMMARY_DAYS", 3)))
    elif isinstance(wx, dict) and wx.get("error"):
        lines.append(f"WEATHER: error={wx['error']}")

    # rag
    rag_hits = results.get("rag") or []
    lines.append(f"RAG_TOPK: {len(rag_hits)} passages retrieved.")
    return "\n".join(lines)

# ------------ main entry ------------

async def answer(req: AskRequest) -> AskResponse:
    # 1) Language
    in_lang = req.lang or detect_lang(req.text)
    text_en = req.text if in_lang == "en" else translate_to_en(req.text)

    # 2) Kick off tasks
    results: Dict[str, Any] = {}
    timings: Dict[str, int] = {}

    tasks: Dict[str, asyncio.Task] = {}
    rag_start = t()
    tasks["rag"] = asyncio.create_task(_run_rag(text_en, k=settings.RAG_TOPK))

    # --- NEW: infer admin area from lat/lon if missing ---
    if req.geo and req.geo.lat is not None and req.geo.lon is not None:
        from app.tools.geo import reverse_geocode_admin
        geo_admin = await reverse_geocode_admin(req.geo.lat, req.geo.lon)
    
        # only fill if user didn’t specify
        if not req.state and geo_admin.get("state"):
            req.state = geo_admin["state"]
        if not req.district and geo_admin.get("district"):
            req.district = geo_admin["district"]
    
    
        if req.market and geo_admin.get("district") and req.district and req.district.lower() != geo_admin["district"].lower():
            req.market = None

    # Weather & Sentinel only if geo present
    weather_start = veg_start = None
    if req.geo and req.geo.lat is not None and req.geo.lon is not None:
        weather_start = t()
        tasks["weather"] = asyncio.create_task(_run_weather(req))
        veg_start = t()
        tasks["veg"] = asyncio.create_task(_run_veg(req))

    # Price and dependent SELL/WAIT (await price first)
    if req.crop or req.state or req.district or req.market:
        price_start = t()
        try:
            price_res = await _run_price(req)
            results["price"] = price_res
        except Exception as e:
            results["price"] = {"error": str(e)}
            price_res = None
        timings["price"] = round((t() - price_start) * 1000)

        # Only start sell/wait if price returned
        if isinstance(price_res, dict) and price_res.get("modal_price_inr_per_qtl") is not None:
            sw_start = t()
            tasks["sell_wait"] = asyncio.create_task(_run_sell_wait(req, price_ctx=price_res, horizon_days=7))
            if settings.SELLWAIT_INCLUDE_2WEEK:
                sw14_start = t()
                tasks["sell_wait_14"] = asyncio.create_task(_run_sell_wait(req, price_ctx=price_res, horizon_days=14))
                timings["_sell14_start_ms"] = round((t() - sw14_start) * 1000)
            timings["_sell_start_ms"] = round((t() - sw_start) * 1000)
            

    # 3) Await remaining tasks robustly
    if "rag" in tasks:
        try:
            results["rag"] = await tasks["rag"]
        except Exception as e:
            results["rag"] = {"error": str(e)}
        timings["rag"] = round((t() - rag_start) * 1000)

    if "weather" in tasks and weather_start is not None:
        try:
            results["weather"] = await tasks["weather"]
        except Exception as e:
            results["weather"] = {"error": str(e)}
        timings["weather"] = round((t() - weather_start) * 1000)

    if "veg" in tasks and veg_start is not None:
        try:
            results["veg"] = await tasks["veg"]
        except Exception as e:
            results["veg"] = {"error": str(e)}
        timings["veg"] = round((t() - veg_start) * 1000)

    if "sell_wait" in tasks:
        sw_done_start = t()
        try:
            results["sell_wait"] = await tasks["sell_wait"]
        except Exception as e:
            results["sell_wait"] = {"error": str(e)}
        timings["sell_wait"] = round((t() - sw_done_start) * 1000)

    if "sell_wait_14" in tasks:
        sw14_done_start = t()
        try:
            results["sell_wait_14"] = await tasks["sell_wait_14"]
        except Exception as e:
            results["sell_wait_14"] = {"error": str(e)}
        timings["sell_wait_14"] = round((t() - sw14_done_start) * 1000)

    results["_timings"] = timings

    # 4) Tool summary + tiny structured facts (for the LLM)
    horizon = req.horizon_days or settings.SELLWAIT_DEFAULT_HORIZON_DAYS

    def _safe_get(d, *keys, default=None):
        cur = d or {}
        for k in keys:
            if not isinstance(cur, dict): return default
            cur = cur.get(k)
        return cur if cur is not None else default

    veg = results.get("veg") or {}
    facts = {
        "price": {
            "commodity": _safe_get(results, "price", "commodity"),
            "today_modal": _safe_get(results, "price", "modal_price_inr_per_qtl"),
            "location": {
                "state": _safe_get(results, "price", "state"),
                "district": _safe_get(results, "price", "district"),
                "market": _safe_get(results, "price", "market")
            }
        },
        "forecast_7d": {
            "expected": _safe_get(results, "sell_wait", "expected_p50_h"),
            "low": _safe_get(results, "sell_wait", "band_p20_h"),
            "high": _safe_get(results, "sell_wait", "band_p80_h"),
            "trend_inr_per_day": _safe_get(results, "sell_wait", "notes", "trend_slope_inr_per_day"),
            "decision": _safe_get(results, "sell_wait", "decision"),
            "confidence": _safe_get(results, "sell_wait", "confidence"),
        },
        "forecast_14d": {
            "expected": _safe_get(results, "sell_wait_14", "expected_p50_h"),
            "low": _safe_get(results, "sell_wait_14", "band_p20_h"),
            "high": _safe_get(results, "sell_wait_14", "band_p80_h"),
            "trend_inr_per_day": _safe_get(results, "sell_wait_14", "notes", "trend_slope_inr_per_day"),
            "decision": _safe_get(results, "sell_wait_14", "decision"),
            "confidence": _safe_get(results, "sell_wait_14", "confidence"),
        },
        "weather_24h": {
            "total_rain_mm": _safe_get(results, "weather", "total_rain_next_24h_mm"),
            "max_temp_c": _safe_get(results, "weather", "max_temp_next_24h_c"),
            "max_wind_ms": _safe_get(results, "weather", "max_wind_next_24h_ms"),
        },
        "weather_next7": {
            "daily": _safe_get(results, "weather", "daily"),
        },
        "veg_indices": {
            "means": {
                "ndvi": veg.get("ndvi_mean"),
                "ndmi": veg.get("ndmi_mean"),
                "ndwi": veg.get("ndwi_mean"),
                "lai":  veg.get("lai_mean"),
            },
            "advice": veg.get("advice") or {},
            "coverage_pct": veg.get("coverage_pct") or {},
            "aoi_m": veg.get("aoi_m"),
            "window_days": veg.get("window_days"),
        },
        "geo": {
            "lat": _safe_get(results, "weather", "lat") or (req.geo.lat if req.geo else None),
            "lon": _safe_get(results, "weather", "lon") or (req.geo.lon if req.geo else None),
        },
        "horizon_days": horizon,
        "language_hint": req.lang or "en"
    }

    import json
    tool_summary = _summarize_tools(results, horizon_days=horizon)
    facts_json = json.dumps(facts, ensure_ascii=False)
    tool_summary = f"{tool_summary}\n\nFACTS_JSON:\n{facts_json}"

    # cap length for LLM prompt
    MAX_CHARS = 6000
    if len(tool_summary) > MAX_CHARS:
        tool_summary = tool_summary[:MAX_CHARS] + " …"

    # 5) Synthesize final answer from RAG + tool summary
    rag_topk = results.get("rag") or []
    ts = t()
    synth = synthesize(text_en, rag_topk, tool_notes=tool_summary)
    results["_timings"]["synthesize_ms"] = round((t() - ts) * 1000)
    ans_en = synth["answer"]
    sources = [Source(**s) for s in synth.get("sources", [])]

    # 6) Translate back if needed
    final_text = ans_en if in_lang == "en" else translate_from_en(ans_en, in_lang)

    return AskResponse(
        answer=final_text,
        sources=sources,            # kept for UI/telemetry; generate.py avoids citing in the text
        tool_notes=results,
        lang=in_lang,
        timings=results.get("_timings", {}),
        debug_info=results if req.debug else None
    )
