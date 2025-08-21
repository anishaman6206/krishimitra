import asyncio
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from app.schemas import AskRequest, AskResponse, Source
from app.tools.lang import detect_lang, translate_to_en, translate_from_en
from app.rag.retrieve import retrieve
from app.rag.generate import synthesize
from app.tools.mandi_cached import latest_price_cached as latest_price, advise_sell_or_wait_cached as advise_sell_or_wait
from app.tools.weather_cached import forecast_24h_cached as wx_forecast
from app.tools.sentinel_cached import ndvi_snapshot_cached as ndvi_snapshot, ndvi_quicklook_cached as ndvi_quicklook
from app.config import settings

def t(): return time.perf_counter()

# ---------- async wrappers for sync calls ----------
async def _translate_to_en_async(text: str) -> str:
    return await asyncio.to_thread(lambda: translate_to_en(text))

async def _translate_from_en_async(text: str, target: str) -> str:
    return await asyncio.to_thread(lambda: translate_from_en(text, target))

async def _run_rag(query_en: str, k: int = 4):
    return await asyncio.to_thread(lambda: retrieve(query_en, k=k))

# ---------- tool runners ----------
async def _run_price(req: AskRequest):
    t0 = t()
    result = await latest_price(
        req.crop or settings.DEFAULT_CROP,  # positional argument
        district=req.district,
        state=req.state,
        market=req.market,
        variety=req.variety,
        grade=req.grade,
        debug=req.debug
    )
    timing_ms = round((t() - t0) * 1000)
    print(f"⏱️  Price lookup: {timing_ms}ms")
    return result

async def _run_sell_wait(req: AskRequest, price_ctx: Optional[Dict[str, Any]] = None):
    t0 = t()
    result = await advise_sell_or_wait(
        commodity=req.crop or settings.DEFAULT_CROP,
        state=req.state, district=req.district, market=req.market,
        variety=req.variety, grade=req.grade,
        horizon_days=req.horizon_days or settings.SELLWAIT_DEFAULT_HORIZON_DAYS,
        qty_qtl=req.qty_qtl,
        debug=req.debug,
        price_context=price_ctx
    )
    timing_ms = round((t() - t0) * 1000)
    print(f"⏱️  Pricing analysis: {timing_ms}ms")
    return result

async def _run_weather(req: AskRequest):
    t0 = t()
    g = req.geo or {}
    if not g or g.lat is None or g.lon is None:
        raise ValueError("weather: missing lat/lon")
    result = await wx_forecast(g.lat, g.lon, tz="auto")
    timing_ms = round((t() - t0) * 1000)
    print(f"⏱️  Weather forecast: {timing_ms}ms")
    return result

async def _run_ndvi(req: AskRequest):
    t0 = t()
    g = req.geo or {}
    if not g or g.lat is None or g.lon is None:
        raise ValueError("ndvi: missing lat/lon")
    result = await ndvi_snapshot(
        lat=g.lat, lon=g.lon,
        aoi_km=settings.SENTINEL_AOI_KM,
        recent_days=settings.SENTINEL_RECENT_DAYS,
        prev_days=settings.SENTINEL_PREV_DAYS,
        gap_days=settings.SENTINEL_GAP_DAYS,
    )
    timing_ms = round((t() - t0) * 1000)
    print(f"⏱️  NDVI analysis: {timing_ms}ms")
    return result

async def _run_ndvi_quicklook(req: AskRequest, aoi_km: Optional[float] = None):
    g = req.geo or {}
    if not g or g.lat is None or g.lon is None:
        raise ValueError("ndvi_quicklook: missing lat/lon")
    img_bytes = await ndvi_quicklook(
        g.lat, g.lon,
        aoi_km=aoi_km or settings.SENTINEL_AOI_KM,
        recent_days=settings.SENTINEL_QUICKLOOK_RECENT_DAYS,
    )
    if not img_bytes:
        return {"available": False}
    fname = f"ndvi_{g.lat:.5f}_{g.lon:.5f}_{(aoi_km or settings.SENTINEL_AOI_KM):.1f}km.png"
    fpath = Path(settings.STATIC_DIR) / fname
    fpath.parent.mkdir(parents=True, exist_ok=True)  # ensure /static exists
    fpath.write_bytes(img_bytes)
    return {"available": True, "url": f"/static/{fname}", "bytes": len(img_bytes)}

# ---------- summarization for tool notes ----------
def _fmt_weather(wx: Dict[str, Any], days_head: int = 3) -> str:
    if not isinstance(wx, dict):
        return "WEATHER: unavailable"
    r = wx.get("total_rain_next_24h_mm")
    t_max = wx.get("max_temp_next_24h_c")
    t_min = wx.get("min_temp_next_24h_c")
    w = wx.get("max_wind_next_24h_ms") or wx.get("max_wind_next_24h_kmh")

    if isinstance(w, (int, float)):
        if "max_wind_next_24h_ms" in wx:
            wtxt = f"{w:.1f} m/s"
        else:
            w_ms = w / 3.6
            wtxt = f"{w_ms:.1f} m/s"
    else:
        wtxt = "—"

    if t_max is not None and t_min is not None:
        temp_txt = f"{t_min:.0f}–{t_max:.0f}°C"
    elif t_max is not None:
        temp_txt = f"max {t_max:.0f}°C"
    else:
        temp_txt = "—"

    line = f"WEATHER 24h: rain≈{r or 0:.0f} mm, temp {temp_txt}, max wind {wtxt}."

    daily = wx.get("daily") or []
    rain_alerts = []
    if isinstance(daily, list) and daily:
        try:
            for i, day in enumerate(daily[:3]):
                rain_mm = day.get("rain_mm") or 0
                rain_chance = day.get("rain_chance_pct") or 0
                date_str = day.get("date", f"Day {i+1}")
                if rain_mm >= 8 or rain_chance >= 70:
                    rain_alerts.append(f"{date_str} ({rain_mm:.1f}mm, {rain_chance:.0f}%)")
            if rain_alerts:
                line += f" RAIN_ALERT: {', '.join(rain_alerts)}."
            tmins = [(d["tmin_c"], d["date"]) for d in daily if d.get("tmin_c") is not None]
            tmaxs = [(d["tmax_c"], d["date"]) for d in daily if d.get("tmax_c") is not None]
            if tmins and tmaxs:
                min_t, min_d = min(tmins, key=lambda x: x[0])
                max_t, max_d = max(tmaxs, key=lambda x: x[0])
                line += f" NEXT 7d: coldest ~{min_t:.0f}°C ({min_d}), hottest ~{max_t:.0f}°C ({max_d})."
        except Exception:
            pass
    return line

def _summarize_tools(results: Dict[str, Any], horizon_days: int) -> str:
    lines: List[str] = []

    # Price
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

    def _last_forecast_triplet(sw: dict):
        fc = sw.get("forecast") or []
        if isinstance(fc, list) and fc:
            last = fc[-1]
            exp  = last.get("p50_adj", last.get("p50"))
            low  = last.get("p20_adj", last.get("p20"))
            high = last.get("p80_adj", last.get("p80"))
            return exp, low, high
        return sw.get("expected_p50_h"), sw.get("band_p20_h"), sw.get("band_p80_h")

    # SELL/WAIT (7d)
    sw = results.get("sell_wait")
    if isinstance(sw, dict) and sw.get("decision"):
        H = horizon_days or settings.SELLWAIT_DEFAULT_HORIZON_DAYS
        span = "WEEK 1" if H == 7 else ("WEEK 2" if H == 14 else f"{H}-DAY")
        notes = sw.get("notes") or {}
        exp, low, high = _last_forecast_triplet(sw)
        thresholds = notes.get("thresholds", {})
        uplift = notes.get("uplift_inr", 0)
        prob_up = notes.get("prob_up", 0)
        trend = notes.get("trend_slope_inr_per_day", 0)
        min_uplift = thresholds.get("min_uplift_inr", 0)
        min_prob = thresholds.get("min_prob_up", 0)
        min_trend = thresholds.get("min_trend_inr_per_day", 0)

        rule_explanation = (
            f"RULES: uplift≥₹{min_uplift}? "
            f"{'YES' if uplift >= min_uplift else 'NO'} (₹{uplift:.1f}), "
            f"prob_up≥{min_prob:.0%}? "
            f"{'YES' if prob_up >= min_prob else 'NO'} ({prob_up:.1%}), "
            f"trend≥₹{min_trend}/day? "
            f"{'YES' if trend >= min_trend else 'NO'} (₹{trend:.1f}/day)."
        )

        line = (
            f"{span}: {'WAIT' if sw['decision']=='WAIT' else 'SELL NOW'} | "
            f"today {_fmt_inr(sw.get('now_price'))}, "
            f"expected {_fmt_inr(exp)}, "
            f"likely range {_fmt_inr(low)}–{_fmt_inr(high)}; "
            f"{_trend_word(notes.get('trend_slope_inr_per_day'))}."
        )
        lines.append(line)
        lines.append(f"DECISION_RULES: {rule_explanation}")
        if isinstance(sw.get("chart"), dict) and sw["chart"].get("url"):
            lines.append(f"FORECAST_IMG: {sw['chart']['url']}")
    elif isinstance(sw, dict) and sw.get("error"):
        lines.append(f"SELL/WAIT: error={sw['error']}")

    # SELL/WAIT (14d optional)
    sw14 = results.get("sell_wait_14")
    if isinstance(sw14, dict) and sw14.get("decision"):
        notes14 = sw14.get("notes") or {}
        exp14, low14, high14 = _last_forecast_triplet(sw14)
        line14 = (
            f"WEEK 2: {'WAIT' if sw14['decision']=='WAIT' else 'SELL NOW'} | "
            f"today {_fmt_inr(sw14.get('now_price'))}, "
            f"expected {_fmt_inr(exp14)}, "
            f"likely range {_fmt_inr(low14)}–{_fmt_inr(high14)}; "
            f"{_trend_word(notes14.get('trend_slope_inr_per_day'))}."
        )
        lines.append(line14)
        if isinstance(sw14.get("chart"), dict) and sw14["chart"].get("url"):
            lines.append(f"FORECAST_IMG_14D: {sw14['chart']['url']}")
    elif isinstance(sw14, dict) and sw14.get("error"):
        lines.append(f"SELL/WAIT_14D: error={sw14['error']}")

    # Weather
    wx = results.get("weather")
    if isinstance(wx, dict) and ("total_rain_next_24h_mm" in wx or "daily" in wx):
        lines.append(_fmt_weather(wx, days_head=settings.WX_SUMMARY_DAYS))
    elif isinstance(wx, dict) and wx.get("error"):
        lines.append(f"WEATHER: error={wx['error']}")

    # NDVI stats
    ndvi = results.get("ndvi")
    if isinstance(ndvi, dict) and ndvi.get("ndvi_latest") is not None:
        cov = ndvi.get("ndvi_coverage_pct")
        used = ndvi.get("aoi_used")
        lines.append(
            "NDVI: mean={:.2f}, prev={}, trend={}, coverage={}%, AOI={} km."
            .format(
                ndvi["ndvi_latest"],
                "NA" if ndvi.get("ndvi_prev") is None else f"{ndvi['ndvi_prev']:.2f}",
                ndvi.get("trend") or "NA",
                "NA" if cov is None else cov,
                "NA" if used is None else used
            )
        )
    elif isinstance(ndvi, dict) and ndvi.get("error"):
        lines.append(f"NDVI: error={ndvi['error']}")

    # NDVI quicklook
    ql = results.get("ndvi_quicklook")
    if isinstance(ql, dict) and ql.get("available"):
        lines.append(f"NDVI_IMG: quicklook available at {ql.get('url')}")
    elif isinstance(ql, dict) and ql.get("available") is False:
        lines.append("NDVI_IMG: unavailable")
    elif isinstance(ql, dict) and ql.get("error"):
        lines.append(f"NDVI_IMG: error={ql['error']}")

    # RAG
    rag_hits = results.get("rag") or []
    lines.append(f"RAG_TOPK: {len(rag_hits)} passages retrieved.")
    return "\n".join(lines)

# ---------- main entry ----------
async def answer(req: AskRequest) -> AskResponse:
    t0 = t()

    # 1) Language handling (non-blocking)
    in_lang = req.lang or detect_lang(req.text)
    text_en = req.text if in_lang == "en" else await _translate_to_en_async(req.text)

    # 2) Fan-out orchestration
    results: Dict[str, Any] = {}
    timings: Dict[str, int] = {}

    tasks = {}
    print("⏳ Starting to await rag...")
    rag_start = t()
    tasks["rag"] = asyncio.create_task(_run_rag(text_en, k=settings.RAG_TOPK))

    weather_start = ndvi_start = None
    print(f"🌦️ Checking weather/NDVI conditions: geo={req.geo}, lat={req.geo.lat if req.geo else None}, lon={req.geo.lon if req.geo else None}")
    if req.geo and req.geo.lat is not None and req.geo.lon is not None:
        print("⏳ Starting to await weather...")
        weather_start = t()
        tasks["weather"] = asyncio.create_task(_run_weather(req))

        print("⏳ Starting to await ndvi...")
        ndvi_start = t()
        tasks["ndvi"] = asyncio.create_task(_run_ndvi(req))
    else:
        print("❌ Weather/NDVI skipped: no geo data")

    # Price first → then sell/wait
    sell_start = sell14_start = None
    print(f"💰 Checking price tool conditions:")
    print(f"   req.crop = '{req.crop}' (bool: {bool(req.crop)})")
    print(f"   req.state = '{req.state}' (bool: {bool(req.state)})")
    print(f"   req.district = '{req.district}' (bool: {bool(req.district)})")
    print(f"   req.market = '{req.market}' (bool: {bool(req.market)})")
    print(f"   Condition result: {bool(req.crop or req.state or req.district or req.market)}")
    
    if req.crop or req.state or req.district or req.market:
        print("⏳ Starting to await price...")
        price_start = t()
        price_res = await _run_price(req)
        price_ms = round((t() - price_start) * 1000)
        results["price"] = price_res
        timings["price"] = price_ms
        print(f"✅ price: completed in {price_ms}ms")

        print("⏳ Starting to await sell_wait...")
        sell_start = t()
        tasks["sell_wait"] = asyncio.create_task(_run_sell_wait(req, price_ctx=price_res))

        if settings.SELLWAIT_INCLUDE_2WEEK:
            print("⏳ Starting to await sell_wait_14...")
            sell14_start = t()
            tasks["sell_wait_14"] = asyncio.create_task(
                advise_sell_or_wait(
                    commodity=req.crop or settings.DEFAULT_CROP,
                    state=req.state, district=req.district, market=req.market,
                    variety=req.variety, grade=req.grade,
                    horizon_days=14, qty_qtl=req.qty_qtl, debug=req.debug,
                    price_context=price_res
                )
            )
    else:
        print("❌ Price tools skipped: no crop/state/district/market data")

    # 3) Await remaining tasks + timings
    if "rag" in tasks:
        try:
            results["rag"] = await tasks["rag"]
            timings["rag"] = round((t() - rag_start) * 1000)
            print("rag: completed")
        except Exception as e:
            timings["rag"] = round((t() - rag_start) * 1000)
            results["rag"] = {"error": str(e)}
            print(f"rag: failed - {str(e)}")

    if "weather" in tasks and weather_start is not None:
        try:
            results["weather"] = await tasks["weather"]
            timings["weather"] = round((t() - weather_start) * 1000)
            print("weather: completed")
        except Exception as e:
            timings["weather"] = round((t() - weather_start) * 1000)
            results["weather"] = {"error": str(e)}
            print(f"weather: failed - {str(e)}")

    if "ndvi" in tasks and ndvi_start is not None:
        try:
            results["ndvi"] = await tasks["ndvi"]
            timings["ndvi"] = round((t() - ndvi_start) * 1000)
            print("ndvi: completed")
        except Exception as e:
            timings["ndvi"] = round((t() - ndvi_start) * 1000)
            results["ndvi"] = {"error": str(e)}
            print(f"ndvi: failed - {str(e)}")

    if "sell_wait" in tasks and sell_start is not None:
        try:
            results["sell_wait"] = await tasks["sell_wait"]
            timings["sell_wait"] = round((t() - sell_start) * 1000)
            print("sell_wait: completed")
        except Exception as e:
            timings["sell_wait"] = round((t() - sell_start) * 1000)
            results["sell_wait"] = {"error": str(e)}
            print(f"sell_wait: failed - {str(e)}")

    if "sell_wait_14" in tasks and sell14_start is not None:
        try:
            results["sell_wait_14"] = await tasks["sell_wait_14"]
            timings["sell_wait_14"] = round((t() - sell14_start) * 1000)
            print("sell_wait_14: completed")
        except Exception as e:
            timings["sell_wait_14"] = round((t() - sell14_start) * 1000)
            results["sell_wait_14"] = {"error": str(e)}
            print(f"sell_wait_14: failed - {str(e)}")

    results["_timings"] = timings

    # 3b) NDVI quicklook after AOI known
    if settings.SENTINEL_ENABLE_QUICKLOOK and req.geo and req.geo.lat is not None and req.geo.lon is not None:
        try:
            aoi_for_img = None
            ndvi_res = results.get("ndvi")
            if isinstance(ndvi_res, dict) and ndvi_res.get("aoi_used"):
                aoi_for_img = float(ndvi_res["aoi_used"])
            results["ndvi_quicklook"] = await _run_ndvi_quicklook(req, aoi_km=aoi_for_img)
        except Exception as e:
            results["ndvi_quicklook"] = {"error": str(e)}

    # 4) Build summarized tool notes + compact facts JSON
    horizon = req.horizon_days or settings.SELLWAIT_DEFAULT_HORIZON_DAYS

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
    facts_json = json.dumps(facts, ensure_ascii=False)

    # Fix percentile monotonicity (safety)
    def _fix_monotone(day):
        p20, p50, p80 = day.get("p20"), day.get("p50"), day.get("p80")
        if None in (p20, p50, p80):
            return day
        original = (p20, p50, p80)
        lo, md, hi = sorted([p20, p50, p80])
        day["p20"], day["p50"], day["p80"] = lo, md, hi
        if original != (lo, md, hi):
            print(f"⚠️  Fixed non-monotonic percentiles: {original} -> ({lo}, {md}, {hi})")
        return day

    for key in ["sell_wait", "sell_wait_14"]:
        forecast = results.get(key, {}).get("forecast", [])
        if isinstance(forecast, list):
            for day in forecast:
                _fix_monotone(day)

    tool_summary = _summarize_tools(results, horizon_days=horizon)
    tool_summary = f"{tool_summary}\n\nFACTS_JSON:\n{facts_json}"

    # 5) Synthesis (sync call)
    rag_topk = results.get("rag") or []
    ts = t()
    synth = synthesize(text_en, rag_topk, tool_notes=tool_summary)
    results["_timings"]["synthesize_ms"] = round((t() - ts) * 1000)
    ans_en = synth["answer"]
    sources = [Source(**s) for s in synth.get("sources", [])]

    # 6) Translate back if needed (non-blocking)
    final_text = ans_en if in_lang == "en" else await _translate_from_en_async(ans_en, in_lang)

    if in_lang != "en" and final_text == ans_en:
        print(f"⚠️  Back-translation skipped or failed; in_lang={in_lang}, ans_en length={len(ans_en)}")
    elif in_lang != "en":
        print(f"✅ Translated from English to {in_lang} (length: {len(ans_en)} -> {len(final_text)})")

    print(f"⏱️  Request timing: {results.get('_timings', {})}")

    return AskResponse(
        answer=final_text,
        sources=sources,
        tool_notes=results,
        lang=in_lang,
        timings=results["_timings"],
        debug_info=results if req.debug else None
    )
