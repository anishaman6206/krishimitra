import os
import datetime as dt
from math import cos, radians
from typing import Dict, Optional

import numpy as np
from sentinelhub import (
    SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions
)
from dotenv import load_dotenv
from pathlib import Path

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
  return {
    input: [{ bands: ["B04", "B08", "SCL"], units: "DN" }],
    output: { bands: 3, sampleType: "AUTO" }
  };
}
function evaluatePixel(s) {
  // mask thick clouds/shadows
  if ([0,3,8,9,10,11].indexOf(s.SCL) !== -1) return [0,0,0];
  let ndvi = (s.B08 - s.B04) / (s.B08 + s.B04 + 1e-6);
  // simple gradient: red (low) -> yellow -> green (high)
  let c = colorBlend(ndvi,
    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    [[0.6,0.0,0.0],[0.8,0.2,0.1],[0.9,0.6,0.1],[0.8,0.9,0.2],[0.4,0.8,0.2],[0.0,0.6,0.0]]
  );
  return c;
}
"""

def _ndvi_quicklook_png(bbox: BBox, start: dt.date, end: dt.date, cfg: SHConfig) -> bytes:
    size = bbox_to_dimensions(bbox, resolution=10)
    req = SentinelHubRequest(
        evalscript=NDVI_VIS_EVALSCRIPT,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=(start.isoformat(), end.isoformat()),
            mosaicking_order="mostRecent",
            other_args={"dataFilter": {"maxCloudCoverage": MAX_CLOUD}}
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=size,
        config=cfg
    )
    return req.get_data()[0]

async def ndvi_quicklook(lat: float, lon: float, aoi_km: float = 0.5,
                         recent_days: int = 20) -> Optional[bytes]:
    cfg = _sh_config()
    bbox = _circle_bbox(lat, lon, aoi_km)

    # try a slightly longer window for imagery (more chance of a clear pixel)
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

def _ndvi_mean_for_range(bbox: BBox, start: dt.date, end: dt.date, cfg: SHConfig) -> Optional[Dict[str, Optional[float]]]:
    size = bbox_to_dimensions(bbox, resolution=10)
    req = SentinelHubRequest(
        evalscript=NDVI_EVALSCRIPT,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=(start.isoformat(), end.isoformat()),
            mosaicking_order="mostRecent",
            other_args={"dataFilter": {"maxCloudCoverage": MAX_CLOUD}}
        )],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=cfg
    )
    arr = req.get_data()[0].squeeze()
    arr = np.array(arr, dtype="float32")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": None,
            "min": None,
            "max": None,
            "median": None
        }
    return {
        "mean": float(np.nanmean(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "median": float(np.nanmedian(arr))
    }

async def ndvi_snapshot(lat: float, lon: float, aoi_km: float = 0.5,
                        recent_days: int = 10, prev_days: int = 10, gap_days: int = 7) -> Dict:
    cfg = _sh_config()
    bbox = _circle_bbox(lat, lon, aoi_km)

    # try up to 3 expansions if cloudy
    def try_stats(length: int, offset: int) -> Dict[str, Optional[float]]:
        start, end = _range(offset, length)
        return _ndvi_mean_for_range(bbox, start, end, cfg)

    cur_len, cur_off = recent_days, 0
    cur_stats = try_stats(cur_len, cur_off)
    for expand in (5, 10):  # widen by +5 then +10 days if needed
        if cur_stats["mean"] is None:
            cur_stats = try_stats(cur_len + expand, cur_off)

    prev_len, prev_off = prev_days, gap_days + recent_days
    prev_stats = try_stats(prev_len, prev_off)
    for expand in (5, 10):
        if prev_stats["mean"] is None:
            prev_stats = try_stats(prev_len + expand, prev_off)

    trend = None
    if cur_stats["mean"] is not None and prev_stats["mean"] is not None:
        delta = cur_stats["mean"] - prev_stats["mean"]
        if   delta > 0.02: trend = "rising"
        elif delta < -0.02: trend = "falling"
        else:               trend = "stable"

    return {
        "ndvi_latest": cur_stats["mean"],
        "ndvi_prev": prev_stats["mean"],
        "trend": trend,
        "bbox_wgs84": [bbox.lower_left, bbox.upper_right],
        "ndvi_min": cur_stats["min"],
        "ndvi_max": cur_stats["max"],
        "ndvi_median": cur_stats["median"],
        "note": f"AOI ~{aoi_km} km radius, res=10 m, windows widened if cloudy (cloud≤{MAX_CLOUD}%)."
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
