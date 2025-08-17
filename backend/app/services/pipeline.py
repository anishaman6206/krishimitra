import asyncio
from typing import List, Dict, Any, Optional

from app.schemas import AskRequest, AskResponse, Source
from app.tools.lang import detect_lang, translate_to_en, translate_from_en
from app.rag.retrieve import retrieve
from app.rag.generate import synthesize
from app.tools.mandi import latest_price
from app.tools.pricing import advise_sell_or_wait
from app.tools.weather import forecast_24h as wx_forecast
from app.tools.sentinel import sentinel_summary  
#from app.tools.sentinel import ndvi_snapshot, ndvi_quicklook
from app.config import settings
from pathlib import Path



# ---------- fan-out helpers ----------

async def _run_rag(query_en: str, k: int = 3):
    # retrieve() is sync; run in a thread for true parallelism
    return await asyncio.to_thread(lambda: retrieve(query_en, k=k))

async def _run_price(req: AskRequest):
    return await latest_price(
        req.crop or settings.DEFAULT_CROP,
        district=req.district, state=req.state, market=req.market,
        variety=req.variety, grade=req.grade, debug=req.debug
    )

async def _run_sell_wait(req: AskRequest):
    return await advise_sell_or_wait(
        commodity=req.crop or settings.DEFAULT_CROP,
        state=req.state, district=req.district, market=req.market,
        variety=req.variety, grade=req.grade,
        horizon_days=req.horizon_days or settings.SELLWAIT_DEFAULT_HORIZON_DAYS,
        qty_qtl=req.qty_qtl,
        debug=req.debug
    )

async def _run_weather(req: AskRequest):
    g = req.geo or {}
    if not g or g.lat is None or g.lon is None:
        raise ValueError("weather: missing lat/lon")
    return await wx_forecast(g.lat, g.lon, tz="auto")



async def _run_veg(req: AskRequest):
    g = req.geo or {}
    if not g or g.lat is None or g.lon is None:
        raise ValueError("veg: missing lat/lon")

    # Read start/max from settings (defaults: start 0.5 km, max 3 km)
    start_km = getattr(settings, "SENTINEL_START_AOI_KM", 0.5)
    max_km   = getattr(settings, "SENTINEL_MAX_AOI_KM", 3.0)

    start_m = int(start_km * 1000)
    max_m   = int(max_km * 1000)

    # Build ordered AOI steps, starting from your chosen start_m and growing up to max_m
    canonical = [200, 500, 1000, 2000, 3000]
    steps = [start_m] + [s for s in canonical if s >= start_m and s <= max_m]
    steps = tuple(dict.fromkeys(steps))  # de-dupe, keep order

    return await sentinel_summary(
        lat=g.lat,
        lon=g.lon,
        farm_size_meters=start_m,                 # starting AOI
        recent_days=getattr(settings, "SENTINEL_RECENT_DAYS", 20),
        resolution=10,
        autogrow=True,                            # ← important
        aoi_steps_m=steps                         # ← seed auto-grow path
    )


# ---------- summarization for tool notes ----------

def _fmt_weather(wx: Dict[str, Any], days_head: int = 3) -> str:
    if not isinstance(wx, dict):
        return "WEATHER: unavailable"

    r = wx.get("total_rain_next_24h_mm", 0)
    t = wx.get("max_temp_next_24h_c")

    if "max_wind_next_24h_kmh" in wx and isinstance(wx["max_wind_next_24h_kmh"], (int, float)):
        w_kmh = float(wx["max_wind_next_24h_kmh"])
    elif "max_wind_next_24h_ms" in wx and isinstance(wx["max_wind_next_24h_ms"], (int, float)):
        w_kmh = float(wx["max_wind_next_24h_ms"]) * 3.6
    else:
        w_kmh = None

    wtxt = f"{w_kmh:.0f} km/h" if isinstance(w_kmh, float) else "—"
    line = f"WEATHER 24h: rain≈{r:.0f} mm, max temp {t if t is not None else '—'}°C, max wind {wtxt}."

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
    if "ndvi" in cov: bits.append(f"NDVI cov {cov['ndvi']}%")
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
    
    def _chance_word(p: float | None) -> str:
        if p is None: return "—"
        return f"{int(round(100*float(p)))}%"
    
    # put this near the other small helpers in pipeline.py
    def _last_forecast_triplet(sw: dict):
        """
        Return (expected, low, high) from the LAST day of the forecast list.
        Prefers *_adj keys; falls back to non-adjusted; finally to top-level fields.
        """
        fc = sw.get("forecast") or []
        if isinstance(fc, list) and fc:
            last = fc[-1]
            exp  = last.get("p50_adj", last.get("p50"))
            low  = last.get("p20_adj", last.get("p20"))
            high = last.get("p80_adj", last.get("p80"))
            return exp, low, high
        # fallback if no per-day forecast
        return sw.get("expected_p50_h"), sw.get("band_p20_h"), sw.get("band_p80_h")

   
    veg = results.get("veg")
    if veg is not None:
        lines.append(_fmt_veg(veg))


 
# ---- SELL/WAIT summaries in plain language (no p50/p80 terms) ----
    sw = results.get("sell_wait")
    if isinstance(sw, dict) and sw.get("decision"):
        H = horizon_days or settings.SELLWAIT_DEFAULT_HORIZON_DAYS
        span = "WEEK 1" if H == 7 else ("WEEK 2" if H == 14 else f"{H}-DAY")
        notes = sw.get("notes") or {}
        exp, low, high = _last_forecast_triplet(sw)  # <-- use last day of forecast
        line = (
            f"{span}: {'WAIT' if sw['decision']=='WAIT' else 'SELL NOW'} | "
            f"today {_fmt_inr(sw.get('now_price'))}, "
            f"expected {_fmt_inr(exp)}, "
            #f"likely range {_fmt_inr(low)}–{_fmt_inr(high)}; "
            f"{_trend_word(notes.get('trend_slope_inr_per_day'))}."
            # f" chance of better price ≈ {_chance_word(notes.get('prob_up'))}; "
            # f" confidence {sw.get('confidence','—')}."
        )
        lines.append(line)
        if isinstance(sw.get("chart"), dict) and sw["chart"].get("url"):
            lines.append(f"FORECAST_IMG: {sw['chart']['url']}")
    elif isinstance(sw, dict) and sw.get("error"):
        lines.append(f"SELL/WAIT: error={sw['error']}")
    
    # Optional second horizon (WEEK 2)
    sw14 = results.get("sell_wait_14")
    if isinstance(sw14, dict) and sw14.get("decision"):
        notes14 = sw14.get("notes") or {}
        exp14, low14, high14 = _last_forecast_triplet(sw14)  # <-- use last day of 14d forecast
        line14 = (
            f"WEEK 2: {'WAIT' if sw14['decision']=='WAIT' else 'SELL NOW'} | "
            f"today {_fmt_inr(sw14.get('now_price'))}, "
            f"expected {_fmt_inr(exp14)}, "
            #f"likely range {_fmt_inr(low14)}–{_fmt_inr(high14)}; "
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
        lines.append(_fmt_weather(wx, days_head=settings.WX_SUMMARY_DAYS))
    elif isinstance(wx, dict) and wx.get("error"):
        lines.append(f"WEATHER: error={wx['error']}")


    # rag
    rag_hits = results.get("rag") or []
    lines.append(f"RAG_TOPK: {len(rag_hits)} passages retrieved.")

    return "\n".join(lines)



# ---------- main entry ----------
async def answer(req: AskRequest) -> AskResponse:
    # 1) Language handling
    in_lang = req.lang or detect_lang(req.text)
    text_en = req.text if in_lang == "en" else translate_to_en(req.text)

    # 2) Fan out (schedule ONCE)
    tasks: Dict[str, asyncio.Future] = {}
    tasks["rag"] = asyncio.create_task(_run_rag(text_en, k=settings.RAG_TOPK))

    # Only run market tools if we have at least some crop/geo hint
    if req.crop or req.state or req.district or req.market:
        tasks["price"] = asyncio.create_task(_run_price(req))
        # primary SELL/WAIT at requested horizon
        tasks["sell_wait"] = asyncio.create_task(_run_sell_wait(req))
        # optional 2-week SELL/WAIT as a second view
        if settings.SELLWAIT_INCLUDE_2WEEK:
            tasks["sell_wait_14"] = asyncio.create_task(
                advise_sell_or_wait(
                    commodity=req.crop or settings.DEFAULT_CROP,
                    state=req.state, district=req.district, market=req.market,
                    variety=req.variety, grade=req.grade,
                    horizon_days=14, qty_qtl=req.qty_qtl, debug=req.debug
                )
            )
    # Weather & Satellite only if geo present
    if req.geo and req.geo.lat is not None and req.geo.lon is not None:
        tasks["weather"] = asyncio.create_task(_run_weather(req))
        tasks["veg"] = asyncio.create_task(_run_veg(req))

    # 3) Gather results robustly
    results: Dict[str, Any] = {}
    for name, fut in tasks.items():
        try:
            results[name] = await fut
        except Exception as e:
            results[name] = {"error": str(e)}

    
    # 4) Build tool summary to guide the LLM
    horizon = req.horizon_days or settings.SELLWAIT_DEFAULT_HORIZON_DAYS

    # 3.5) Build structured facts from tool outputs (keep it tiny & generic)
    import json
    
    def _safe_get(d, *keys, default=None):
        cur = d or {}
        for k in keys:
            if not isinstance(cur, dict): return default
            cur = cur.get(k)
        return cur if cur is not None else default
    
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
            "chance_up": _safe_get(results, "sell_wait", "notes", "prob_up"),
            "decision": _safe_get(results, "sell_wait", "decision"),
            "confidence": _safe_get(results, "sell_wait", "confidence"),
        },
        "forecast_14d": {
            "expected": _safe_get(results, "sell_wait_14", "expected_p50_h"),
            "low": _safe_get(results, "sell_wait_14", "band_p20_h"),
            "high": _safe_get(results, "sell_wait_14", "band_p80_h"),
            "trend_inr_per_day": _safe_get(results, "sell_wait_14", "notes", "trend_slope_inr_per_day"),
            "chance_up": _safe_get(results, "sell_wait_14", "notes", "prob_up"),
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
        "ndvi": {
            "mean": _safe_get(results, "ndvi", "ndvi_latest"),
            "prev": _safe_get(results, "ndvi", "ndvi_prev"),
            "trend": _safe_get(results, "ndvi", "trend"),
            "coverage_pct": _safe_get(results, "ndvi", "ndvi_coverage_pct"),
            "aoi_km": _safe_get(results, "ndvi", "aoi_used"),
        },
        "geo": {
            "lat": _safe_get(results, "weather", "lat") or (req.geo.lat if req.geo else None),
            "lon": _safe_get(results, "weather", "lon") or (req.geo.lon if req.geo else None),
        },
        "horizon_days": req.horizon_days or settings.SELLWAIT_DEFAULT_HORIZON_DAYS,
        "language_hint": req.lang or "en"
    }

    veg = results.get("veg") or {}
    facts["veg_indices"] = {
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
    }

    facts["veg_indices"] = {
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
    }

    facts_json = json.dumps(facts, ensure_ascii=False)

    MAX_CHARS = 6000
    tool_summary = _summarize_tools(results, horizon_days=horizon)
    tool_summary = f"{tool_summary}\n\nFACTS_JSON:\n{facts_json}"
    if len(tool_summary) > MAX_CHARS:
        tool_summary = tool_summary[:MAX_CHARS] + " …"


    # 5) Synthesize final answer from RAG + tool summary
    rag_topk = results.get("rag") or []
    synth = synthesize(text_en, rag_topk, tool_notes=tool_summary)
    ans_en = synth["answer"]
    sources = [Source(**s) for s in synth.get("sources", [])]

    # 6) Translate back if needed
    final_text = ans_en if in_lang == "en" else translate_from_en(ans_en, in_lang)

    # 7) Follow-ups
    followups = [
        "Ask this in my language",
        "Compare another market or crop",
        "Change holding period",
        "Share my location for local weather/NDVI" if not (req.geo and req.geo.lat is not None) else "See field-wise NDVI history",
    ]

    return AskResponse(
        answer=final_text,
        sources=sources,
        tool_notes=results,   # raw tool outputs for debugging/telemetry
        follow_ups=followups,
        lang=in_lang,
    )

