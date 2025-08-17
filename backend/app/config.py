# backend/app/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path(__file__).parents[3] / '.env'
load_dotenv(dotenv_path)


class Settings:
    # --- OpenAI ---
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    # --- Data.gov.in (mandi) ---
    DATA_GOV_IN_API_KEY: str = os.getenv("DATA_GOV_IN_API_KEY", "")

    # --- Sentinel Hub ---
    SH_CLIENT_ID: str     = os.getenv("SH_CLIENT_ID", "")
    SH_CLIENT_SECRET: str = os.getenv("SH_CLIENT_SECRET", "")

    # --- Pipeline knobs ---
    DEFAULT_CROP: str = os.getenv("DEFAULT_CROP", "Tomato")
    RAG_TOPK: int = int(os.getenv("RAG_TOPK", "4"))

    # Sell/Wait defaults
    SELLWAIT_DEFAULT_HORIZON_DAYS: int = int(os.getenv("SELLWAIT_DEFAULT_HORIZON_DAYS", "7"))

    # Weather
    WX_FORECAST_DAYS: int = int(os.getenv("WX_FORECAST_DAYS", "7"))
    WX_SUMMARY_DAYS: int  = int(os.getenv("WX_SUMMARY_DAYS", "3"))

    STATIC_DIR: str = os.getenv("STATIC_DIR", str((Path(__file__).resolve().parents[1] / "static")))
    SENTINEL_ENABLE_QUICKLOOK: bool = os.getenv("SENTINEL_ENABLE_QUICKLOOK", "1") == "1"
    SENTINEL_QUICKLOOK_RECENT_DAYS: int = int(os.getenv("SENTINEL_QUICKLOOK_RECENT_DAYS", "20"))

    # Sentinel/NDVI windows
 
    SENTINEL_PREV_DAYS: int       = int(os.getenv("SENTINEL_PREV_DAYS", "10"))
    SENTINEL_GAP_DAYS: int        = int(os.getenv("SENTINEL_GAP_DAYS", "7"))

    SENTINEL_START_AOI_KM: float = 0.5    # start small
    SENTINEL_MAX_AOI_KM: float = 3.0      # grow up to 3 km
    SENTINEL_RECENT_DAYS: int = 20


    SELLWAIT_MIN_UPLIFT_INR: int = 15 # require at least ₹15/qtl upside to WAIT
# or a percent version: SELLWAIT_MIN_UPLIFT_PCT = 0.01  # 1%
    SELLWAIT_MIN_PROB: float = 0.65
    SELLWAIT_MIN_TREND_INR_PER_DAY: float = 5

    # Also compute a 2-week advisory alongside the primary horizon
    SELLWAIT_INCLUDE_2WEEK: bool = bool(int(os.getenv("SELLWAIT_INCLUDE_2WEEK", "1")))


    # Cache
    CACHE_TTL_SEC: int = 900

settings = Settings()
