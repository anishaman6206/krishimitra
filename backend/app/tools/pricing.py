# backend/app/tools/pricing.py
import os
import json
import asyncio
import datetime as dt
import time
from typing import Optional, Dict, Any, List


import numpy as np
import pandas as pd
from joblib import load

from app.tools.mandi import latest_price  # uses data.gov.in current-day endpoint

# --------------------------------------------------------------------------------------
# Config / Artifacts
# --------------------------------------------------------------------------------------
GLOBAL_MODEL_DIR = os.getenv(
    "PRICE_GLOBAL_MODEL_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "models", "pricing_global")
)
# small widen factor for bands using validation residual std (tunable)
PRICE_BAND_WIDEN_K = float(os.getenv("PRICE_BAND_WIDEN_K", "0.2"))

# lazy caches
_M20 = _M50 = _M80 = _META = _ENC = None


def _artifact_path(name: str) -> str:
    return os.path.join(GLOBAL_MODEL_DIR, name)


def _load_artifacts():
    """Load p20/p50/p80 LightGBM models, meta.json, and encoder.json (once)."""
    global _M20, _M50, _M80, _META, _ENC
    if _M50 is None:
        _M20 = load(_artifact_path("model_p20.joblib"))
        _M50 = load(_artifact_path("model_p50.joblib"))
        _M80 = load(_artifact_path("model_p80.joblib"))
        with open(_artifact_path("meta.json"), "r") as f:
            _META = json.load(f)
        # encoder is optional but recommended
        enc_path = _artifact_path("encoder.json")
        if os.path.exists(enc_path):
            with open(enc_path, "r", encoding="utf-8") as f:
                _ENC = json.load(f)
        else:
            _ENC = {
                "commodity": {},
                "state": {},
                "district": {},
                "market": {},
                "variety": {},
                "grade": {}
            }
    return _M20, _M50, _M80, _META, _ENC


# --------------------------------------------------------------------------------------
# Encoding + feature building
# --------------------------------------------------------------------------------------
def _encode_id(enc_map: Dict[str, int], key: Optional[str]) -> int:
    """Map a category value to its integer ID using exported encoder; unknowns -> 0."""
    if key is None:
        return 0
    return int(enc_map.get(str(key), 0))


def _make_feat_row_from_hist(
    hist_df: pd.DataFrame,
    fdate: pd.Timestamp,
    ids: Dict[str, int]
) -> Dict[str, float]:
    """
    Build a single feature row for fdate using the current history (date, modal_price).
    We only have current-day price from API; we backfill lags/rolls with last value.
    """
    tail = hist_df.sort_values("date")["modal_price"].astype(float)
    # if empty history (shouldn't happen), return zeros
    if len(tail) == 0:
        tail = pd.Series([0.0])

    row = {
        "doy": float(fdate.timetuple().tm_yday),
        "dow": float(fdate.weekday()),
        "month": float(fdate.month),
        # categorical ids
        "commodity_id": float(ids.get("commodity_id", 0)),
        "state_id": float(ids.get("state_id", 0)),
        "district_id": float(ids.get("district_id", 0)),
        "market_id": float(ids.get("market_id", 0)),
        "variety_id": float(ids.get("variety_id", 0)),
        "grade_id": float(ids.get("grade_id", 0)),
    }
    # lags (backfill with last available)
    for L in (1, 7, 14, 28):
        row[f"lag_{L}"] = float(tail.iloc[-1]) if len(tail) < L else float(tail.iloc[-L])

    # rolling windows (backfill with last few where not enough)
    for W in (7, 14, 28):
        tw = tail.iloc[-W:] if len(tail) >= W else tail
        row[f"rollmean_{W}"] = float(tw.mean())
        row[f"rollstd_{W}"] = float(tw.std() if len(tw) > 1 else 0.0)

    return row


def _adjust_quantiles(p50: float, p20: float, p80: float) -> tuple[float, float, float]:
    """
    Bias-correct all quantiles using global residual median, and widen bands slightly.
    We shift by resid_median (calibration) and then widen by +/- k * resid_std.
    """
    _, _, _, meta, _ = _load_artifacts()
    shift = float(meta.get("resid_median", 0.0) or 0.0)
    std = float(meta.get("resid_std", 0.0) or 0.0)

    p50a = p50 + shift
    p20a = p20 + shift - PRICE_BAND_WIDEN_K * std
    p80a = p80 + shift + PRICE_BAND_WIDEN_K * std

    # monotonic cleanup
    mid = p50a
    lo = min(p20a, mid)
    hi = max(p80a, mid)
    return mid, lo, hi


def _forecast_horizon_from_global_models(
    now_price: float,
    now_date: dt.date,
    ids: Dict[str, int],
    horizon_days: int,
) -> List[Dict[str, Any]]:
    """
    Recursive forecast for H days using global p20/p50/p80 models.
    Seed history with one point: (now_date, now_price).
    """
    m20, m50, m80, meta, _ = _load_artifacts()
    features: List[str] = meta.get("features", [])

    # seed history with current price (single point)
    hist = pd.DataFrame([{"date": pd.to_datetime(now_date), "modal_price": float(now_price)}])

    out: List[Dict[str, Any]] = []
    last_date = pd.to_datetime(now_date)

    for d in range(1, horizon_days + 1):
        fdate = last_date + pd.Timedelta(days=d)
        row = _make_feat_row_from_hist(hist, fdate, ids)
        Xf = pd.DataFrame([row])

        # ensure feature alignment (add missing = 0.0, drop extras)
        for col in features:
            if col not in Xf.columns:
                Xf[col] = 0.0
        Xf = Xf[features]

        # raw quantiles
        y20 = float(m20.predict(Xf)[0])
        y50 = float(m50.predict(Xf)[0])
        y80 = float(m80.predict(Xf)[0])

        # calibrated + widened
        y50a, y20a, y80a = _adjust_quantiles(y50, y20, y80)

        out.append({
            "date": fdate.date().isoformat(),
            "p20": int(round(y20)),
            "p50": int(round(y50)),
            "p80": int(round(y80)),
            "p20_adj": int(round(y20a)),
            "p50_adj": int(round(y50a)),
            "p80_adj": int(round(y80a)),
        })

        # recursive: append adjusted median as pseudo-observation
        hist = pd.concat([hist, pd.DataFrame([{"date": fdate, "modal_price": y50a}])], ignore_index=True)

    return out

def _approx_prob_above(now: float, p20: float, p50: float, p80: float) -> float:
    """
    Heuristic P(price > now) using p20/p50/p80.
    - ≤p20 -> ~0.8
    - =p50 -> 0.5
    - ≥p80 -> ~0.2
    Linear in-between.
    """
    if p80 <= p20:  # degenerate band
        return 0.5 if now <= p50 else 0.4
    if now <= p20:
        return 0.8
    if now >= p80:
        return 0.2
    if now <= p50:
        span = max(1.0, p50 - p20)
        return 0.8 - 0.3 * (now - p20) / span   # 0.8 -> 0.5
    # now in (p50, p80)
    span = max(1.0, p80 - p50)
    return 0.5 - 0.3 * (now - p50) / span       # 0.5 -> 0.2


# --------------------------------------------------------------------------------------
# Public decision function
# --------------------------------------------------------------------------------------
async def advise_sell_or_wait(
    commodity: str,
    state: Optional[str] = None,
    district: Optional[str] = None,
    market: Optional[str] = None,
    variety: Optional[str] = None,
    grade: Optional[str] = None,
    horizon_days: int = 7,
    qty_qtl: Optional[float] = None,  # kept for signature compatibility; not used
    debug: bool = False,
    price_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Decision using ONLY:
      - current-day price (from data.gov.in via latest_price),
      - global quantile models p20/p50/p80 (calibrated),
    No carrying cost / interest / shrinkage. Pure price-expectation + band risk.

    Decision rule:
      - Compute p50_adj for H days ahead.
      - If p50_adj <= now_price → SELL_NOW (expected gain not positive).
      - Else WAIT, and report risk via p20_adj (downside) and p80_adj (upside).
      - Confidence heuristic from band width relative to price.
    """
    t0 = time.perf_counter()
    
    # 1) Current-day price row - use injected price if available
    if price_context and isinstance(price_context, dict):
        price_row = price_context
        print(f"[pricing] Using provided price context (no fetch)")
    else:
        # FALLBACK: fetch price (prefer cached version)
        print(f"[pricing] Fetching price (no context provided)")
        price_row = await latest_price(
            commodity=commodity,
            state=state, district=district, market=market,
            variety=variety, grade=grade,
            cache_ttl_ok=True, debug=debug
        )
    
    now_price = float(price_row["modal_price_inr_per_qtl"])
    # prefer API date string if present; else today
    adate = price_row.get("arrival_date")
    try:
        now_date = dt.datetime.strptime(adate, "%Y-%m-%d").date() if adate else dt.date.today()
    except Exception:
        now_date = dt.date.today()

    # 2) Build categorical IDs using exported encoder
    _, _, _, _, enc = _load_artifacts()
    ids = {
        "commodity_id": _encode_id(enc.get("commodity", {}), commodity),
        "state_id": _encode_id(enc.get("state", {}), state),
        "district_id": _encode_id(enc.get("district", {}), district),
        "market_id": _encode_id(enc.get("market", {}), market),
        "variety_id": _encode_id(enc.get("variety", {}), variety),
        "grade_id": _encode_id(enc.get("grade", {}), grade),
    }

    # 3) Forecast horizon
    fc = _forecast_horizon_from_global_models(
        now_price=now_price,
        now_date=now_date,
        ids=ids,
        horizon_days=horizon_days
    )

    
    if not fc:
        # fallback: no forecast → default to SELL_NOW neutral response
        return {
            "decision": "SELL_NOW",
            "now_price": now_price,
            "expected_p50_h": now_price,
            "band_p20_h": now_price,
            "band_p80_h": now_price,
            "confidence": 0.5,
            "context": price_row,
            "notes": {"reason": "no_forecast"}
        }

    last = fc[-1]
    p50_h = float(last["p50_adj"])
    p20_h = float(last["p20_adj"])
    p80_h = float(last["p80_adj"])

    MIN_UPLIFT_INR = float(os.getenv("SELLWAIT_MIN_UPLIFT_INR", "25"))   # minimum ₹ upside to WAIT
    MIN_PROB_UP   = float(os.getenv("SELLWAIT_MIN_PROB", "0.60"))        # need ≥60% chance of upside
    MIN_TREND_DPD = float(os.getenv("SELLWAIT_MIN_TREND_INR_PER_DAY", "0.0"))  # non-negative slope


    # --- Decision with thresholds & heuristics ---
    uplift = p50_h - now_price

    # p50 path slope over horizon (momentum)
    p50_path = [float(d.get("p50_adj", d.get("p50"))) for d in fc]
    slope = (p50_path[-1] - p50_path[0]) / max(1, (len(p50_path) - 1))  # INR/day

    # probability of finishing above current price (heuristic from quantiles)
    prob_up = _approx_prob_above(now_price, p20_h, p50_h, p80_h)

    # gate: need enough expected upside, non-negative trend, and decent probability
    wait = (uplift >= MIN_UPLIFT_INR) and (slope >= MIN_TREND_DPD) and (prob_up >= MIN_PROB_UP)
    decision = "WAIT" if wait else "SELL_NOW"

    # Confidence: combine band tightness + prob_up (bounded 0.4..0.9)
    band_width = max(1.0, p80_h - p20_h)
    rel_band = band_width / max(1.0, now_price)
    confidence = float(np.clip(0.55 + 0.5*(prob_up - 0.5) - 0.25*rel_band, 0.4, 0.9))


    # 6) Optional value impact if qty is provided
    value_impact = None
    if qty_qtl is not None:
        uplift = max(0.0, p50_h - now_price)
        value_impact = int(round(uplift * float(qty_qtl)))

    # 7) Assemble result
    result = {
        "decision": decision,
        "now_price": now_price,
        "expected_p50_h": p50_h,
        "band_p20_h": p20_h,
        "band_p80_h": p80_h,
        "confidence": round(confidence, 2),
        "context": price_row,
        "forecast": fc,
        "notes": {
            "horizon_days": horizon_days,
            "uplift_inr": round(float(uplift), 2),
            "prob_up": round(float(prob_up), 3),
            "trend_slope_inr_per_day": round(float(slope), 2),
            "rel_band": rel_band,
            "thresholds": {
                "min_uplift_inr": MIN_UPLIFT_INR,
                "min_prob_up": MIN_PROB_UP,
                "min_trend_inr_per_day": MIN_TREND_DPD
            }
        },
        "value_impact_for_qty": value_impact
    }

    if debug:
        print("[pricing] ids=", ids)
        print("[pricing] decision pack=", json.dumps(result, indent=2, ensure_ascii=False))
    
    timing_ms = round((time.perf_counter() - t0) * 1000)
    print(f"⏱️  Pricing analysis: {timing_ms}ms")
    return result


# --------------------------------------------------------------------------------------
# CLI for quick testing
# --------------------------------------------------------------------------------------
async def _cli(args):
    try:
        out = await advise_sell_or_wait(
            commodity=args.commodity,
            state=args.state, district=args.district, market=args.market,
            variety=args.variety, grade=args.grade,
            horizon_days=args.horizon,
            qty_qtl=args.qty,
            debug=args.debug
        )
        print(json.dumps(out, indent=2, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"error": str(e)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Price advisory using global LightGBM quantile models (p20/p50/p80).")
    parser.add_argument("--commodity", required=True)
    parser.add_argument("--state", default=None)
    parser.add_argument("--district", default=None)
    parser.add_argument("--market", default=None)
    parser.add_argument("--variety", default=None)
    parser.add_argument("--grade", default=None)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--qty", type=float, default=None, help="Optional quantity in quintals for value impact")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    asyncio.run(_cli(args))
