KrishiMitra AI — Farmer Copilot (Local Dev Guide)

A unified assistant for Indian farmers that brings together:

Market prices (Agmarknet / data.gov.in) + ML forecasting (p20/p50/p80 quantiles) and a farmer-friendly Sell / Wait decision for week-1 and week-2.

Weather (7-day summary + 24-hour details).

Satellite vegetation (Sentinel-2 NDVI/NDMI/NDWI/LAI), with auto-grown AOI and human-readable advice.

RAG over your PDF/TXT seeds (LangChain + Chroma + OpenAI).

Telegram bot with slot-filling UX, voice (STT/TTS), and leaf disease detector (ViT) powered by a local model + LLM-based care tips.

1) Quick Start
# 1) Clone
git clone <your-repo-url> && cd krishimitra

# 2) Python env
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 3) Install deps
pip install -r requirements.txt

# 4) Create .env (see template below)
copy .env.example .env   # (Windows)
# or
cp .env.example .env     # (macOS/Linux)
# then fill values

# 5) Build RAG index 
python backend/app/rag/index.py --rebuild

# 6) Run backend API (default: http://127.0.0.1:8000)
uvicorn backend.app.main:app --reload

# 7) Run Telegram bot (in another terminal)
python bots/telegram/bot.py

2) What’s Inside (Architecture)
backend/
  app/
    services/pipeline.py   ← orchestrates tools + RAG → final answer
    tools/
      mandi.py             ← data.gov.in (Agmarknet) client + caching
      pricing.py           ← quantile LightGBM (p20/p50/p80) + SELL/WAIT
      weather.py           ← 24h + 7d forecast summary
      sentinel.py          ← Sentinel Hub indices (NDVI/NDMI/NDWI/LAI)
    rag/
      index.py             ← Chroma index build/load
      retrieve.py          ← top-k passages
      generate.py          ← LLM synthesis (tool+RAG aware)
    utils/cache.py         ← simple in-memory TTL cache
bots/
  telegram/bot.py          ← chat, slot filling, voice, disease photos
vit_disease/
  vit_model.py             ← local ViT classifier (leaf disease)
  llm_helper.py            ← care advice text via LLM
models/
  pricing_global/          ← p20/p50/p80 models + meta + encoder (you add)


High-level flow:

User asks → bot sends payload to /ask.

pipeline fans out: price → pricing (sell/wait), weather, vegetation (Sentinel), RAG, then synthesizes one grounded answer.

Bot renders a short, practical reply with optional buttons (NDVI image, horizon 7/14, etc.).

For a leaf photo, bot runs ViT → gets a label → asks LLM for localized care guidance → replies.

3) Environment (.env)

Use .env.example as a base. Fill these:

# --- OpenAI (RAG + generation) ---
OPENAI_API_KEY=sk-...
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini

# --- Agmarknet (data.gov.in) ---
DATA_GOV_IN_API_KEY=your_api_key

# --- Sentinel Hub ---
SH_CLIENT_ID=your_client_id
SH_CLIENT_SECRET=your_client_secret

# --- Telegram ---
TELEGRAM_BOT_TOKEN=123456:ABC...

# --- Backend URL (for bot -> backend) ---
BACKEND_URL=http://127.0.0.1:8000

# --- Pricing model location (optional if using default) ---
PRICE_GLOBAL_MODEL_DIR=backend/models/pricing_global

# --- Voice (optional) ---
# If you use Sarvam or any TTS/STT provider, add their keys here
SARVAM_API_KEY=...

# --- Misc knobs (optional defaults exist) ---
SELLWAIT_DEFAULT_HORIZON_DAYS=7
SELLWAIT_INCLUDE_2WEEK=true
SENTINEL_START_AOI_KM=0.5
SENTINEL_MAX_AOI_KM=3.0
SENTINEL_RECENT_DAYS=45
WX_SUMMARY_DAYS=7


4) Required Local Artifacts
Pricing models (global quantile regressors)

Place under backend/models/pricing_global/:

model_p20.joblib
model_p50.joblib
model_p80.joblib
meta.json             # feature names etc.
encoder.json          # category encodings (commodity/state/district/market/variety/grade)

ViT plant disease model

Place under vit_disease/vit-plant-disease/:

config.json
preprocessor_config.json
model.safetensors
training_args.bin      # optional

5) Run & Test Locally
5.1 Backend API

Health

curl -s http://127.0.0.1:8000/health


Ask (full featured)

curl -s -X POST http://127.0.0.1:8000/ask \
  -H 'content-type: application/json' \
  -d '{
    "text": "Should I sell tomatoes now?",
    "crop": "Tomato",
    "state": "Karnataka",
    "district": "Bangalore",
    "market": "Ramanagara",
    "geo": {"lat": 12.522, "lon": 76.897},
    "horizon_days": 7,
    "debug": true
  }' | jq .


You should see a single, grounded answer plus tool_notes (prices, weather, vegetation summaries) and timings.

Put PDFs/TXT in backend/ingestion/seeds/.

Build index: python backend/app/rag/index.py --rebuild.

Now questions like “Which tomato variety for monsoon in Karnataka?” will incorporate your seeds.

5.2 Telegram bot

Run:

python bots/telegram/bot.py


Try these:

Share location (📍) — enables local weather & vegetation.

Set market quickly:

Set market: Karnataka, Bangalore, Ramanagara


Set crop / horizon:

Set crop tomato
Set horizon 14


Ask:

“Should I sell tomatoes now?”

“Kal barish hogi? irrigation kab karu?”

“Can I wait 2 weeks for a better price?”

“Analyze NDVI for my field.”

Send a leaf photo — you’ll get Prediction + care tips.

Voice note — ask by voice; tap “Hear this answer” to receive TTS audio.

6) How decisions are made (farmer-friendly)

Sell/Wait:
Uses 3 quantile models p20 / p50 / p80 to forecast price for each day.
We don’t expose p-terms to the farmer; instead:

Week-1 and Week-2 summaries show today’s price, the expected (last day’s p50), and a simple trend (rising / falling / flat).

Final decision is “WAIT” if expected improves enough vs today; otherwise “SELL NOW”.

Vegetation advice (Sentinel):
We compute NDVI/NDMI/NDWI/LAI over the last ~45 days, auto-growing AOI up to 3 km to beat clouds.
Advice uses clear thresholds, e.g.:

NDVI > 0.6 → “Vegetation very healthy.”

NDMI < 0 → “Clear water stress; irrigate soon.”

NDWI < −0.3 → “Surface looks dry; plan irrigation.”

Weather:
24-hour rain / max temp / max wind, plus the coldest & hottest day in the next 7 days.

RAG + Tools synthesis:
The LLM sees a compact tool summary + a structured FACTS_JSON. Prompting keeps it practical (no p50 jargon; short actions first).

7) Typical User Journeys to Test

“Should I sell tomatoes now?”

If market is set → uses live price + forecast + weather context.

If market unknown → bot asks Set market:, then answers.

“When should I irrigate?”

Uses NDMI/NDWI + rain forecast → “Irrigate lightly tomorrow” or “No irrigation needed”.

“Will next week’s temperature drop kill my yield?”

Uses min/max temps for 7 days + vegetation trend → practical risk statement + precaution.

“What seed variety suits this unpredictable weather?”

Uses RAG (your PDFs) + local weather and vegetation status → 2-3 line, concrete suggestion.

“Where can I get affordable credit?”

Answers with KCC and relevant scheme hints (from seeds), avoids jargon.

“Analyze NDVI data for insights”

Returns 2–4 bullet summary (vegetation health, leaf moisture, soil moisture, canopy) + reliability note if coverage is low.

Leaf photo

Returns ViT disease label + LLM care text (localized language).

8) Configuration knobs (useful during testing)

In .env (or app/config/settings.py if you keep constants there):

SELLWAIT_DEFAULT_HORIZON_DAYS (7)

SELLWAIT_INCLUDE_2WEEK (true/false)

PRICE_BAND_WIDEN_K (0.2) — widens forecast range slightly for safety

SENTINEL_START_AOI_KM (0.5), SENTINEL_MAX_AOI_KM (3.0), SENTINEL_RECENT_DAYS (45)

WX_SUMMARY_DAYS (7)

PRICE_GLOBAL_MODEL_DIR path

Caches (in-memory TTL):

Prices cached ~24h

Weather ~6h

Vegetation ~7d

Flush programmatically if needed:

from backend.app.utils.cache import init_cache  # already called on startup
# to clear: restart the process (in-memory), or expose a small admin endpoint if desired

9) Troubleshooting

FFmpeg not found (voice):
Install FFmpeg and ensure ffmpeg.exe is on PATH. Close & re-open terminal. ffmpeg -version should work.

Telegram “Conflict: terminated by other getUpdates request”:
You have another instance of the bot running. Stop it and run one process only.

Sentinel: “No valid pixels / coverage = 0%”:
Clouds! Try larger AOI (auto-grow already up to 3 km) or increase SENTINEL_RECENT_DAYS to 60.

Agmarknet data missing for your district/market:
We progressively relax filters (state → commodity). If you must stay in-state, keep strict_state=True (default) and set a nearby market manually.

ViT model path error:
Ensure vit_disease/vit-plant-disease/ contains config.json, preprocessor_config.json, model.safetensors.

Pricing models not found:
Put model_p20.joblib, model_p50.joblib, model_p80.joblib, meta.json, encoder.json into backend/models/pricing_global/ or set PRICE_GLOBAL_MODEL_DIR.

10) API Reference (minimal)

GET /health → { "ok": true }

POST /ask → body:

{
  "text": "Should I sell tomatoes now?",
  "lang": "en",
  "crop": "Tomato",
  "state": "Karnataka",
  "district": "Bangalore",
  "market": "Ramanagara",
  "geo": {"lat": 12.522, "lon": 76.897},
  "horizon_days": 7,
  "debug": true
}


Response: { answer, sources, tool_notes, timings }

11) Notes & Limitations

Forecasts are guidance, not guarantees. We avoid heavy jargon (p50/p80) and present farmer-friendly text.

Vegetation metrics are averages over an AOI; they’re best for trend and relative status, not precise plant-level diagnosis.

Voice (STT/TTS) depends on your chosen provider (e.g., Sarvam). If you skip it, the bot still works with text & photos.

12) Next steps (optional)

BentoML service for packaging backend + disease API.

WhatsApp channel via Twilio using the same backend.

Price chart image in bot buttons for week-1/week-2 (optional visual).

.env.example
OPENAI_API_KEY=
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini

DATA_GOV_IN_API_KEY=

SH_CLIENT_ID=
SH_CLIENT_SECRET=

TELEGRAM_BOT_TOKEN=
BACKEND_URL=http://127.0.0.1:8000

# optional
PRICE_GLOBAL_MODEL_DIR=backend/models/pricing_global
SELLWAIT_DEFAULT_HORIZON_DAYS=7
SELLWAIT_INCLUDE_2WEEK=true
SENTINEL_START_AOI_KM=0.5
SENTINEL_MAX_AOI_KM=3.0
SENTINEL_RECENT_DAYS=45
WX_SUMMARY_DAYS=7

# voice (optional)
SARVAM_API_KEY=
