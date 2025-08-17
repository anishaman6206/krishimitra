import os
import datetime as dt
import time
from math import cos, radians
from typing import Dict, Optional
from PIL import Image
from io import BytesIO

import numpy as np
from sentinelhub import (
    SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions
)

from dotenv import load_dotenv
from pathlib import Path

def t(): return time.perf_counter()

dotenv_path = Path(__file__).parents[3] / '.env'
load_dotenv(dotenv_path)


from typing import Tuple

MAX_CLOUD = 70  # % cap for scene cloud coverage

def _range(days_back: int, length: int) -> Tuple[dt.date, dt.date]:
    end = dt.date.today() - dt.timedelta(days=days_back)
    start = end - dt.timedelta(days=length)
    return start, end

# --- Sentinel Hub config via OAuth ---
def _sh_config() -> SHConfig:
    cfg = SHConfig()
    cfg.sh_client_id = os.getenv("SH_CLIENT_ID")
    cfg.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
    if not cfg.sh_client_id or not cfg.sh_client_secret:
        raise RuntimeError("Missing SH_CLIENT_ID/SH_CLIENT_SECRET in environment")
    return cfg

# --- Build a small bbox around (lat, lon) using a radius in km ---
def _circle_bbox(lat: float, lon: float, radius_km: float) -> BBox:
    # Convert km to degrees (approx). Good enough for small AOIs (<= 2–3 km).
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * cos(radians(lat))
    dlat = (radius_km * 1000.0) / meters_per_deg_lat
    dlon = (radius_km * 1000.0) / meters_per_deg_lon
    return BBox([lon - dlon, lat - dlat, lon + dlon, lat + dlat], crs=CRS.WGS84)


NDVI_VIS_EVALSCRIPT = """
//VERSION=3
function setup() {
  return { input: [{ bands: ["B04","B08","SCL"], units: "DN" }], output: { bands: 3, sampleType: "AUTO" } };
}
function evaluatePixel(s) {
  if ([0,3,8,9,10,11].indexOf(s.SCL) !== -1) return [0,0,0];
  let ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-6);
  let c = colorBlend(ndvi,
    [0.0,0.2,0.4,0.6,0.8,1.0],
    [[0.6,0.0,0.0],[0.8,0.2,0.1],[0.9,0.6,0.1],[0.8,0.9,0.2],[0.4,0.8,0.2],[0.0,0.6,0.0]]
  );
  return c;
}
"""

def _encode_png(arr: np.ndarray) -> bytes:
    # arr may be float 0..1 or uint8; ensure uint8 [0..255], 3 channels
    if arr.dtype.kind == "f":
        arr = np.clip(arr, 0.0, 1.0) * 255.0
    arr = arr.astype("uint8")
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.stack([arr]*3, axis=-1)
    img = Image.fromarray(arr)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def _ndvi_quicklook_png(bbox: BBox, start: dt.date, end: dt.date, cfg: SHConfig) -> bytes:
    size = bbox_to_dimensions(bbox, resolution=10)
    req = SentinelHubRequest(
        evalscript=NDVI_VIS_EVALSCRIPT,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=(start.isoformat(), end.isoformat()),
            mosaicking_order="leastCC",  # <-- prefer least cloud cover
            other_args={"dataFilter": {"maxCloudCoverage": MAX_CLOUD}}
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=size,
        config=cfg
    )
    # SentinelHub Python client returns a decoded numpy array for PNG
    arr = req.get_data()[0]
    return _encode_png(arr)

async def ndvi_quicklook(lat: float, lon: float, aoi_km: float = 0.5, recent_days: int = 20) -> Optional[bytes]:
    cfg = _sh_config()
    bbox = _circle_bbox(lat, lon, aoi_km)
    start, end = dt.date.today() - dt.timedelta(days=recent_days), dt.date.today()
    try:
        return _ndvi_quicklook_png(bbox, start, end, cfg)
    except Exception:
        return None
 
    


# --- Evalscript: NDVI with basic cloud class filtering (SCL) ---
NDVI_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B08", "SCL"], units: "DN" }],
    output: { bands: 1, sampleType: "FLOAT32" }
  };
}
// SCL values (simplified): 0 NODATA, 3 SC_CLOUD_SHADOW, 8 CLOUD_MEDIUM_PROB, 9 CLOUD_HIGH_PROB, 10 THIN_CIRRUS, 11 SNOW
function evaluatePixel(s) {
  if (s.SCL == 0 || s.SCL == 3 || s.SCL == 8 || s.SCL == 9 || s.SCL == 10 || s.SCL == 11) return [NaN];
  let ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-6);
  return [ndvi];
}
"""

def _ndvi_mean_for_range(bbox, start, end, cfg):
    size = bbox_to_dimensions(bbox, resolution=10)
    req = SentinelHubRequest(
        evalscript=NDVI_EVALSCRIPT,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=(start.isoformat(), end.isoformat()),
            mosaicking_order="leastCC",
            other_args={"dataFilter": {"maxCloudCoverage": MAX_CLOUD}}
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox, size=size, config=cfg
    )
    arr_all = np.array(req.get_data()[0]).squeeze().astype("float32")
    total = arr_all.size
    arr = arr_all[np.isfinite(arr_all)]
    used = arr.size
    if used == 0:
        return {"mean": None, "min": None, "max": None, "median": None, "coverage_pct": 0.0}
    return {
        "mean": float(np.nanmean(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "median": float(np.nanmedian(arr)),
        "coverage_pct": round(100.0 * used / max(1, total), 2),
    }


async def ndvi_snapshot(lat: float, lon: float, aoi_km: float = 0.5,
                        recent_days: int = 10, prev_days: int = 10, gap_days: int = 7) -> Dict:
    start = t()
    cfg = _sh_config()

    # AOIs to try in order (don’t duplicate if caller already passed a large one)
    aoi_tries = [round(aoi_km, 3)]
    for grow in (1.0, 2.0, 3.0):
        if grow > aoi_km + 1e-9:
            aoi_tries.append(grow)

    attempts = []
    aoi_used = None
    cur_stats_best: Dict[str, Optional[float]] | None = None
    prev_stats_best: Dict[str, Optional[float]] | None = None
    bbox_used: BBox | None = None

    for aoi_try in aoi_tries:
        bbox = _circle_bbox(lat, lon, aoi_try)

        def try_stats(length: int, offset: int) -> Dict[str, Optional[float]]:
            start, end = _range(offset, length)
            return _ndvi_mean_for_range(bbox, start, end, cfg)

        # current window (strict) with widening
        cur_stats = try_stats(recent_days, 0)
        for expand in (5, 10):
            if cur_stats["mean"] is None:
                cur_stats = try_stats(recent_days + expand, 0)

        # previous window (strict) with widening
        prev_stats = try_stats(prev_days, gap_days + recent_days)
        for expand in (5, 10):
            if prev_stats["mean"] is None:
                prev_stats = try_stats(prev_days + expand, gap_days + recent_days)

        attempts.append({
            "aoi_km": aoi_try,
            "cur_mean": cur_stats.get("mean"),
            "cur_min": cur_stats.get("min"),
            "cur_max": cur_stats.get("max"),
            "cur_median": cur_stats.get("median"),
            "cur_cov_pct": cur_stats.get("coverage_pct"),
            "prev_mean": prev_stats.get("mean"),
        })
        
        # Accept only if we have a real signal (not a flat/all-zero image)
        def _usable(stats: Dict[str, Optional[float]]) -> bool:
            if stats.get("mean") is None:
                return False
            cov = float(stats.get("coverage_pct") or 0.0)
            minv = stats.get("min"); maxv = stats.get("max"); medv = stats.get("median")
            # reject clearly degenerate all-zero frames
            if minv == 0 and maxv == 0 and medv == 0:
                return False
            # reject near-flat frames (no contrast)
            if minv is not None and maxv is not None and (maxv - minv) < 0.01:
                return False
            # ensure at least some usable coverage (tweak threshold if you like)
            if cov < 1.0:
                return False
            return True
        
        if _usable(cur_stats):
            aoi_used = aoi_try
            bbox_used = bbox
            cur_stats_best = cur_stats
            prev_stats_best = prev_stats
            break
# else: continue loop to try a larger AOI

    # If none of the AOIs produced valid pixels, return transparent result for the last AOI
    if aoi_used is None:
        bbox_last = _circle_bbox(lat, lon, aoi_tries[-1])
        return {
            "ndvi_latest": None,
            "ndvi_prev": None,
            "trend": None,
            "bbox_wgs84": [bbox_last.lower_left, bbox_last.upper_right],
            "ndvi_min": None,
            "ndvi_max": None,
            "ndvi_median": None,
            "ndvi_coverage_pct": 0.0,
            "note": f"AOI auto-grow tried {aoi_tries} km; clouds≤{MAX_CLOUD}%.",
            "aoi_used": None,
            "attempts": attempts,
        }

    # Compute trend if both windows are available
    trend = None
    if cur_stats_best.get("mean") is not None and prev_stats_best.get("mean") is not None:
        delta = float(cur_stats_best["mean"]) - float(prev_stats_best["mean"])
        if   delta > 0.02: trend = "rising"
        elif delta < -0.02: trend = "falling"
        else:               trend = "stable"

    total_ms = round((t() - start) * 1000)
    print(f"⏱️  NDVI snapshot: {total_ms}ms (AOI: {aoi_used}km)")

    return {
        "ndvi_latest": cur_stats_best.get("mean"),
        "ndvi_prev": prev_stats_best.get("mean"),
        "trend": trend,
        "bbox_wgs84": [bbox_used.lower_left, bbox_used.upper_right],
        "ndvi_min": cur_stats_best.get("min"),
        "ndvi_max": cur_stats_best.get("max"),
        "ndvi_median": cur_stats_best.get("median"),
        "ndvi_coverage_pct": cur_stats_best.get("coverage_pct"),
        "note": f"AOI auto-grow used {aoi_used} km; res=10 m; clouds≤{MAX_CLOUD}%.",
        "aoi_used": aoi_used,
        "attempts": attempts,
    }

if __name__ == '__main__':
    import asyncio

    # Print the environment variables
    sh_client_id = os.getenv("SH_CLIENT_ID")
    sh_client_secret = os.getenv("SH_CLIENT_SECRET")
    print("SH_CLIENT_ID:", sh_client_id)
    print("SH_CLIENT_SECRET:", sh_client_secret)

    result = asyncio.run(ndvi_snapshot(52.0, 13.0))  # sample lat/lon values
    print("NDVI Snapshot:", result)
    print("NDVI Quicklook:", asyncio.run(ndvi_quicklook(52.0, 13.0)))
