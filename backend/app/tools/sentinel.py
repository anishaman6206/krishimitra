import os
import asyncio
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import datetime as dt
from app.config import settings

import numpy as np
from sentinelhub import BBox, CRS, SentinelHubRequest, DataCollection, MimeType, SHConfig
from dotenv import load_dotenv

# --- Configuration ---

# Load environment variables from the root of the project (.env file)
dotenv_path = Path(__file__).parents[3] / '.env'
load_dotenv(dotenv_path)

# Sentinel Hub Credentials (must be in your .env file)
# These are required for fetching on-demand satellite imagery and indices.
SH_CLIENT_ID = os.getenv("SH_CLIENT_ID")
SH_CLIENT_SECRET = os.getenv("SH_CLIENT_SECRET")

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

# --- Helper Functions ---

def _configure_sentinel_hub():
    """Configures the Sentinel Hub API credentials."""
    if not SH_CLIENT_ID or not SH_CLIENT_SECRET:
        raise RuntimeError("SH_CLIENT_ID or SH_CLIENT_SECRET not set in .env file")
    config = SHConfig()
    config.sh_client_id = SH_CLIENT_ID
    config.sh_client_secret = SH_CLIENT_SECRET
    config.save()
    return config

def get_bounding_box(lat: float, lon: float, size_meters: int = 100) -> BBox:
    """Creates a square bounding box around a central lat/lon point."""
    meters_per_degree = 111320 
    lat_span = size_meters / meters_per_degree
    lon_span = size_meters / (meters_per_degree * np.cos(np.radians(lat)))
    min_lat, max_lat = lat - lat_span / 2, lat + lat_span / 2
    min_lon, max_lon = lon - lon_span / 2, lon + lon_span / 2
    return BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)

# --- Sentinel Hub Data Function ---

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
    image_width = max(8, int(farm_size_meters / resolution))
    image_height = max(8, int(farm_size_meters / resolution))

    # Build only requests for products we know
    reqs: List[Tuple[str, SentinelHubRequest]] = []
    for name in products:
        script = AVAILABLE_SENTINEL_PRODUCTS.get(name)
        if not script:
            continue
        reqs.append((
            name,
            SentinelHubRequest(
                evalscript=script,
                input_data=[SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=date_range,
                    mosaicking_order="leastCC"  # prefer least cloud-cover scene
                )],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=bbox,
                size=(image_width, image_height),
                config=config
            )
        ))

    if not reqs:
        return {}

    loop = asyncio.get_running_loop()
    data_list = await loop.run_in_executor(None, lambda: [r.get_data() for _, r in reqs])

    results: Dict[str, Dict[str, Any]] = {}
    for (name, _), data in zip(reqs, data_list):
        if not data:
            continue
        arr = np.asarray(data[0], dtype="float32")
        # Expect single-band arrays for indices
        if arr.ndim >= 2:
            finite = np.isfinite(arr)
            cov = float(finite.mean() * 100.0) if arr.size else 0.0
            vals = arr[finite]
            stats = {
                "mean": float(np.nanmean(vals)) if vals.size else None,
                "std_dev": float(np.nanstd(vals)) if vals.size else None,
                "min": float(np.nanmin(vals)) if vals.size else None,
                "max": float(np.nanmax(vals)) if vals.size else None,
            }
            results[name] = {"stats": stats, "coverage_pct": round(cov, 2)}
    return results


# --- Main Integrated Function ---

async def get_comprehensive_sentinel_data(
    lat: float, 
    lon: float,
    date_range: Tuple[str, str] = ("2025-08-01", "2025-08-15"),
    farm_size_meters: int = 100
) -> Dict[str, Any]:
    """
    This is the main function for this module.
    It fetches a suite of data products from Sentinel Hub to provide a holistic view of a farm.
    """
    sentinel_products_to_fetch = ["ndvi", "ndmi", "ndwi", "lai"]
    
    sentinel_results = await get_sentinel_data(lat, lon, sentinel_products_to_fetch, date_range, farm_size_meters, 10)
    
    # Synthesize the final report
    final_report = {
        "location": {"lat": lat, "lon": lon},
        "sentinel_data": {}
    }
    
    for product, result in sentinel_results.items():
        if result:
            final_report["sentinel_data"][product] = {
                "stats": result.get("stats"),
            
            }
            
    return final_report



async def sentinel_summary(
    lat: float,
    lon: float,
    farm_size_meters: int = 200,   # starting AOI edge (~square)
    recent_days: int = 45,         # INCREASED: Look back further to find clear skies
    resolution: int = 10,
    autogrow: bool = True,
    aoi_steps_m: tuple[int, ...] = (200, 500, 1000, 2000, 3000),
    min_cov_pct: float = 10.0,     # require at least this % valid pixels
    low_coverage_threshold: int = 50 # NEW: Warn if coverage is below this %
) -> Dict[str, Any]:
    """
    Returns compact means for NDVI/NDMI/NDWI/LAI with coverage%.
    If autogrow, tries larger AOIs up to 3 km until any index has coverage ≥ min_cov_pct.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=recent_days)
    date_range = (start.isoformat(), end.isoformat())

    tried: list[dict] = []
    aoi_list = aoi_steps_m if autogrow else (farm_size_meters,)

    chosen_data: Dict[str, Any] | None = None
    aoi_used_m: int | None = None

    for aoi_m in aoi_list:
        data = await get_sentinel_data(
            lat=lat,
            lon=lon,
            products=["ndvi", "ndmi", "ndwi", "lai"],
            date_range=date_range,
            farm_size_meters=aoi_m,
            resolution=resolution
        )

        # Collect quick coverage signal
        cov = {k: (v or {}).get("coverage_pct", 0.0) for k, v in (data or {}).items()}
        any_good = any((isinstance(cov.get(k), (int, float)) and cov.get(k, 0.0) >= min_cov_pct)
                       for k in ("ndvi", "ndmi", "ndwi", "lai"))

        tried.append({"aoi_m": aoi_m, "coverage_pct": cov})

        if any_good:
            chosen_data = data
            aoi_used_m = aoi_m
            break

    # If nothing met coverage threshold, fall back to the last attempt (might be empty)
    if chosen_data is None:
        chosen_data = data  # last loop value
        aoi_used_m = aoi_list[-1]

    def _stat(name: str, key: str) -> Optional[float]:
        d = chosen_data.get(name) or {}
        s = d.get("stats") or {}
        return s.get(key)

    cov = {k: (v or {}).get("coverage_pct", 0.0) for k, v in (chosen_data or {}).items()}
    avg_coverage = np.mean([c for c in cov.values() if c > 0]) if any(c > 0 for c in cov.values()) else 0.0


    out = {
        "ndvi_mean": _stat("ndvi", "mean"),
        "ndmi_mean": _stat("ndmi", "mean"),
        "ndwi_mean": _stat("ndwi", "mean"),
        "lai_mean":  _stat("lai",  "mean"),
        "coverage_pct": cov,          # per-index coverage
        "aoi_m": aoi_used_m,
        "window_days": recent_days,
        "attempts": tried,            # useful for debugging
        # NEW: Add a reliability flag for the AI
        "reliability": "low" if avg_coverage < low_coverage_threshold else "high"
    }

    # Attach farmer-friendly advice (uses your helper)
    out["advice"] = _interpret_satellite_simple(
        out.get("ndvi_mean"), out.get("ndmi_mean"),
        out.get("ndwi_mean"), out.get("lai_mean")
    )

    # Short, human summary for LLM/tool-notes (optional)
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

    return out

# --- Farmer-friendly interpretation based on your thresholds ---

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



# --- Command-Line Interface for Testing ---
if __name__ == "__main__":
    import argparse
    import asyncio

    def _fmt(x, nd=4):
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "N/A"

    parser = argparse.ArgumentParser(description="Sentinel farm indices quick report")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--aoi_m", type=int, default=200, help="Starting AOI edge in meters")
    parser.add_argument("--days", type=int, default=45, help="Lookback window in days")
    parser.add_argument("--res", type=int, default=10, help="Pixel resolution in meters")
    parser.add_argument("--no-autogrow", action="store_true", help="Disable AOI auto-grow (stays at --aoi_m)")
    args = parser.parse_args()

    async def _run():
        rep = await sentinel_summary(
            lat=args.lat,
            lon=args.lon,
            farm_size_meters=args.aoi_m,
            recent_days=args.days,
            resolution=args.res,
            autogrow=not args.no_autogrow,
        )

        cov = rep.get("coverage_pct") or {}
        advice = (rep.get("advice") or {}).get("summary_lines") or []
        reliability = rep.get("reliability")

        print("\n--- Comprehensive Sentinel Hub Farm Data Report ---")
        print(f"AOI used: {rep.get('aoi_m')} m   Window: {rep.get('window_days')} d   Res: {args.res} m\n")

        print(f"Sentinel Avg NDVI: {_fmt(rep.get('ndvi_mean'))}   (cov {cov.get('ndvi','N/A')}%)")
        print(f"Sentinel Avg NDMI: {_fmt(rep.get('ndmi_mean'))}   (cov {cov.get('ndmi','N/A')}%)")
        print(f"Sentinel Avg NDWI: {_fmt(rep.get('ndwi_mean'))}   (cov {cov.get('ndwi','N/A')}%)")
        print(f"Sentinel Avg LAI:  {_fmt(rep.get('lai_mean'))}   (cov {cov.get('lai','N/A')}%)")
        print("--------------------------------------------------\n")
        
        # NEW: Add a reliability warning to the output
        if reliability == 'low' and any(v > 0 for v in cov.values()):
            print("Warning: Results are based on low data coverage and may be less reliable.\n")


        if advice:
            for line in advice[:4]:
                print(f"- {line}")
        else:
            # Helpful hint if nothing valid
            if all((cov.get(k, 0) == 0 for k in ("ndvi", "ndmi", "ndwi", "lai"))):
                print("Note: No valid pixels found (likely due to persistent clouds/shadows). Tried up to 3 km AOI and a 45-day window.")
            else:
                print("Note: Limited signal. Consider increasing --days.")

    asyncio.run(_run())
