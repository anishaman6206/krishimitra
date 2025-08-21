import os
import asyncio
import datetime as dt
import time
from math import cos, radians
from typing import Dict, Optional, List, Tuple, Any
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

MAX_CLOUD = 70  # % cap for scene cloud coverage

def _range(days_back: int, length: int) -> Tuple[dt.date, dt.date]:
    end = dt.date.today() - dt.timedelta(days=days_back)
    start = end - dt.timedelta(days=length)
    return start, end

# --- Sentinel Hub EVALSCRIPTS ---
# These are small JavaScript snippets that run on Sentinel Hub's servers to process the raw satellite data.

# B02=Blue, B03=Green, B04=Red, B08=NIR, B11=SWIR1
# Each band captures a different part of the light spectrum.

EVALSCRIPT_NDVI = """
//VERSION=3
function setup() { return { input: ["B04","B08","SCL"], output: { bands: 1, sampleType: "FLOAT32" } }; }
function evaluatePixel(s) {
  if ([0,3,8,9,10,11].indexOf(s.SCL) !== -1) return [NaN];
  return [(s.B08 - s.B04) / (s.B08 + s.B04 + 1e-6)];
}
"""

EVALSCRIPT_NDMI = """
//VERSION=3
function setup() { return { input: ["B08","B11","SCL"], output: { bands: 1, sampleType: "FLOAT32" } }; }
function evaluatePixel(s) {
  if ([0,3,8,9,10,11].indexOf(s.SCL) !== -1) return [NaN];
  return [(s.B08 - s.B11) / (s.B08 + s.B11 + 1e-6)];
}
"""

EVALSCRIPT_NDWI = """
//VERSION=3
function setup() { return { input: ["B03","B08","SCL"], output: { bands: 1, sampleType: "FLOAT32" } }; }
function evaluatePixel(s) {
  if ([0,3,8,9,10,11].indexOf(s.SCL) !== -1) return [NaN];
  return [(s.B03 - s.B08) / (s.B03 + s.B08 + 1e-6)];
}
"""

EVALSCRIPT_LAI = """
//VERSION=3
function setup() { return { input: ["B04","B08","SCL"], output: { bands: 1, sampleType: "FLOAT32" } }; }
function evaluatePixel(s) {
  if ([0,3,8,9,10,11].indexOf(s.SCL) !== -1) return [NaN];
  let ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-6);
  let lai = 0.57 * Math.exp(2.33 * ndvi); // empirical approx
  return [lai];
}
"""

# Dictionary to map product names to their scripts for easy access.
AVAILABLE_SENTINEL_PRODUCTS = {
    "ndvi": EVALSCRIPT_NDVI,
    "ndmi": EVALSCRIPT_NDMI,
    "ndwi": EVALSCRIPT_NDWI,
    "lai": EVALSCRIPT_LAI,
}

# --- Sentinel Hub config via OAuth ---
def _sh_config() -> SHConfig:
    cfg = SHConfig()
    cfg.sh_client_id = os.getenv("SH_CLIENT_ID")
    cfg.sh_client_secret = os.getenv("SH_CLIENT_SECRET")
    if not cfg.sh_client_id or not cfg.sh_client_secret:
        raise RuntimeError("Missing SH_CLIENT_ID/SH_CLIENT_SECRET in environment")
    return cfg

def _configure_sentinel_hub():
    """Configures the Sentinel Hub API credentials."""
    return _sh_config()

def get_bounding_box(lat: float, lon: float, size_meters: int = 100) -> BBox:
    """Creates a square bounding box around a central lat/lon point."""
    meters_per_degree = 111320 
    lat_span = size_meters / meters_per_degree
    lon_span = size_meters / (meters_per_degree * np.cos(np.radians(lat)))
    min_lat, max_lat = lat - lat_span / 2, lat + lat_span / 2
    min_lon, max_lon = lon - lon_span / 2, lon + lon_span / 2
    return BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)

# --- Build a small bbox around (lat, lon) using a radius in km ---
def _circle_bbox(lat: float, lon: float, radius_km: float) -> BBox:
    # Convert km to degrees (approx). Good enough for small AOIs (<= 2–3 km).
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * cos(radians(lat))
    dlat = (radius_km * 1000.0) / meters_per_deg_lat
    dlon = (radius_km * 1000.0) / meters_per_deg_lon
    return BBox([lon - dlon, lat - dlat, lon + dlon, lat + dlat], crs=CRS.WGS84)

# --- New Comprehensive Sentinel Hub Data Functions ---

async def get_sentinel_data(
    lat: float,
    lon: float,
    products: List[str],
    date_range: Tuple[str, str],
    farm_size_meters: int,
    resolution: int
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch selected single-band indices and return compact stats + coverage_pct.
    No image arrays are returned to keep payloads small.
    """
    config = _configure_sentinel_hub()
    bbox = get_bounding_box(lat, lon, farm_size_meters)

    # Ensure reasonable pixel dims (avoid too small tiles)
    image_width = max(8, int(farm_size_meters / max(1, resolution)))
    image_height = max(8, int(farm_size_meters / max(1, resolution)))

    # Build only requests for products we know
    valid_products = []
    reqs: List[Tuple[str, SentinelHubRequest]] = []
    for name in products or []:
        script = AVAILABLE_SENTINEL_PRODUCTS.get(name)
        if not script:
            continue
        valid_products.append(name)
        reqs.append((
            name,
            SentinelHubRequest(
                evalscript=script,
                input_data=[SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=date_range,
                    mosaicking_order="leastCC",  # prefer least cloud-cover scene
                    other_args={"dataFilter": {"maxCloudCoverage": MAX_CLOUD}}
                )],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=bbox,
                size=(image_width, image_height),
                config=config
            )
        ))

    if not reqs:
        # None of the requested products are supported
        raise ValueError(f"No valid Sentinel products requested. Supported: {sorted(AVAILABLE_SENTINEL_PRODUCTS.keys())}")

    # Run blocking get_data() calls off the event loop
    loop = asyncio.get_running_loop()
    data_list = await loop.run_in_executor(None, lambda: [r.get_data() for _, r in reqs])

    results: Dict[str, Dict[str, Any]] = {}
    for (name, _), data in zip(reqs, data_list):
        if not data:
            continue
        arr = np.asarray(data[0], dtype="float32")

        # Expect single-band arrays; handle corner cases
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        elif arr.ndim > 2:
            # Unexpected shape; skip to be safe
            continue

        if arr.size == 0:
            continue

        finite = np.isfinite(arr)
        cov = float(finite.mean() * 100.0) if arr.size else 0.0
        vals = arr[finite]
        if vals.size == 0:
            continue

        stats = {
            "mean": float(np.nanmean(vals)),
            "std_dev": float(np.nanstd(vals)),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
        }
        results[name] = {"stats": stats, "coverage_pct": round(cov, 2)}

    return results


async def sentinel_summary(
    lat: float,
    lon: float,
    farm_size_meters: int = 200,
    recent_days: int = 20,
    resolution: int = 10
) -> Dict[str, Any]:
    """
    Get comprehensive satellite summary with farmer-friendly advice.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=recent_days)
    date_range = (start.isoformat(), end.isoformat())

    data = await get_sentinel_data(
        lat=lat, lon=lon,
        products=["ndvi", "ndmi", "ndwi", "lai"],
        date_range=date_range,
        farm_size_meters=farm_size_meters,
        resolution=resolution
    )

    def _stat(name: str, key: str) -> Optional[float]:
        d = data.get(name) or {}
        s = d.get("stats") or {}
        return s.get(key)

    out = {
        "ndvi_mean": _stat("ndvi", "mean"),
        "ndmi_mean": _stat("ndmi", "mean"),
        "ndwi_mean": _stat("ndwi", "mean"),
        "lai_mean":  _stat("lai",  "mean"),
        "coverage_pct": {k: v.get("coverage_pct") for k, v in data.items()},
        "aoi_m": farm_size_meters,
        "window_days": recent_days,
    }

    # Attach farmer-friendly advice
    out["advice"] = _interpret_satellite_simple(
        out["ndvi_mean"], out["ndmi_mean"],
        out["ndwi_mean"], out["lai_mean"]
    )

    # Optional one-line summary for the LLM prompt/tool_notes
    bullets = []
    if out["ndvi_mean"] is not None:
        ndvi = out["ndvi_mean"]
        bullets.append("vegetation healthy" if ndvi >= 0.4 else ("vegetation fair" if ndvi >= 0.2 else "sparse vegetation"))
    if out["ndmi_mean"] is not None:
        ndmi = out["ndmi_mean"]
        bullets.append("leaf moisture good" if ndmi >= 0.1 else ("moderate moisture" if ndmi >= 0.0 else "drying leaves"))
    if out["ndwi_mean"] is not None:
        ndwi = out["ndwi_mean"]
        bullets.append("soil looks wet" if ndwi > 0.1 else ("soil moisture adequate" if ndwi >= -0.3 else "surface dry"))
    if out["lai_mean"] is not None:
        bullets.append(f"canopy ~{out['lai_mean']:.1f} LAI")
    out["summary"] = ", ".join(bullets) if bullets else None

    # If nothing usable came back (e.g., persistent clouds), surface a helpful message
    if not any(v for k, v in data.items() if isinstance(v, dict) and v.get("stats", {}).get("mean") is not None):
        return {
            "error": "No usable satellite signal in the requested window (likely clouds). "
                     f"Try increasing recent_days or farm_size_meters. Window={recent_days}d, size={farm_size_meters}m"
        }

    return out

# --- Farmer-friendly interpretation based on thresholds ---

def _interpret_satellite_simple(ndvi: float | None,
                                ndmi: float | None,
                                ndwi: float | None,
                                lai:  float | None) -> dict:
    lines = []
    out = {}

    # NDVI (vegetation vigor)
    if ndvi is not None:
        if ndvi > 0.6:
            out["ndvi_band"] = "excellent"
            lines.append(f"Vegetation looks very healthy (NDVI {ndvi:.2f}).")
        elif 0.3 <= ndvi <= 0.6:
            out["ndvi_band"] = "moderate"
            lines.append(f"Vegetation is in a normal growth stage (NDVI {ndvi:.2f}).")
        elif 0.1 <= ndvi < 0.3:
            out["ndvi_band"] = "potential_stress"
            lines.append(f"Signs of potential stress (NDVI {ndvi:.2f}); check pests/nutrients.")
        else:  # < 0.1
            out["ndvi_band"] = "bare_or_unhealthy"
            lines.append(f"Very low green cover (NDVI {ndvi:.2f}).")

    # NDMI (leaf water content / stress)
    if ndmi is not None:
        if ndmi > 0.2:
            out["ndmi_band"] = "no_stress"
            lines.append("Leaves have good moisture; no water stress.")
        elif 0.0 <= ndmi <= 0.2:
            out["ndmi_band"] = "mild_stress"
            lines.append("Early signs of water stress; watch irrigation schedule.")
        else:  # < 0.0
            out["ndmi_band"] = "high_stress"
            lines.append("Clear water stress; irrigate soon to avoid yield loss.")

    # NDWI (surface/soil moisture → irrigation)
    if ndwi is not None:
        if ndwi > 0.1:
            out["ndwi_band"] = "wet"
            lines.append("Soil looks wet; no irrigation needed now.")
        elif -0.3 <= ndwi <= 0.1:
            out["ndwi_band"] = "adequate"
            lines.append("Soil moisture seems adequate for now.")
        else:  # < -0.3
            out["ndwi_band"] = "dry"
            lines.append("Soil surface looks dry; plan irrigation soon.")

    # LAI (canopy density / stage)
    if lai is not None:
        if lai > 2.5:
            out["lai_band"] = "full_canopy"
            lines.append(f"Full, healthy canopy (LAI ~{lai:.1f}).")
        elif 1.0 <= lai <= 2.5:
            out["lai_band"] = "moderate_canopy"
            lines.append(f"Canopy developing (LAI ~{lai:.1f}).")
        else:
            out["lai_band"] = "sparse_canopy"
            lines.append(f"Sparse leaf cover (LAI ~{lai:.1f}).")

    # High-level irrigation nudge from NDWI/NDMI combo
    irrigation = None
    if ndwi is not None and ndmi is not None:
        if ndwi < -0.3 or ndmi < 0.0:
            irrigation = "IRRIGATE_SOON"
        elif ndwi > 0.1 and ndmi > 0.2:
            irrigation = "NO_IRRIGATION"
        else:
            irrigation = "MONITOR"
    out["irrigation_hint"] = irrigation

    out["summary_lines"] = lines
    return out


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
    arr = np.clip(arr, 0, 255).astype("uint8")
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.stack([arr]*3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    img = Image.fromarray(arr, mode="RGB")
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
