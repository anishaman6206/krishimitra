# KrishiMitra AI — The Farmer's Copilot 🇮🇳

A unified, data-driven assistant for Indian farmers, designed to provide actionable, real-time advice on the most critical aspects of agriculture.

This project brings together a suite of powerful tools to answer a farmer's most pressing questions, from market prices to crop health, all through a simple, accessible interface.

---

## ✨ Core Features

- **📈 Market Price Forecasting:** Fetches live prices from Agmarknet (data.gov.in) and uses ML to provide a farmer-friendly Sell / Wait recommendation for the next 1-2 weeks.
- **🛰️ Real-Time Satellite Analysis:** Leverages Sentinel-2 data to analyze vegetation health (NDVI, NDMI, NDWI, LAI), automatically adjusting the area of interest to find cloud-free images.
- **🌦️ Hyperlocal Weather Forecasts:** Provides detailed 7-day and 24-hour weather summaries for the farmer's specific location.
- **📚 Fact-Grounded RAG:** Uses a Retrieval-Augmented Generation pipeline over your own documents (PDFs/TXT) to answer complex queries without hallucination.
- **🌍 Location Intelligence:**
  - `geocode.py`: Converts location names (e.g., "Kharagpur") into precise latitude and longitude using OpenStreetMap.
  - `geo.py`: Performs reverse geocoding to identify the state, district, etc., from geographic coordinates.
- **🗣️ Multilingual Support:**
  - `lang.py`: Automatically detects the user's language and provides answers in the same language for a natural, intuitive experience.
- **🌿 Leaf Disease Detection:** A local Vision Transformer (ViT) model identifies common plant diseases from a photo, with the LLM providing tailored care advice.
- **🤖 Accessible Bots:** Deployed via a Telegram bot with a user-friendly interface, including voice commands (STT/TTS) and quick-action buttons.

---

## 🚀 Quick Start

### Clone the Repository

```bash
git clone <your-repo-url> && cd krishimitra
```

### Set Up Python Environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create and Configure .env File

```bash
# On Windows
copy .env.example .env
# On macOS/Linux
cp .env.example .env
```

Now, open the `.env` file and fill in your secret API keys.

### Build the RAG Index

```bash
python backend/app/rag/index.py --rebuild
```

### Run the Backend API

```bash
uvicorn backend.app.main:app --reload
```

The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

### Run the Telegram Bot (in a new terminal)

```bash
python bots/telegram/bot.py
```

---

## 🏗️ What’s Inside (Architecture)

```
backend/
  app/
    services/pipeline.py    # Orchestrates tools + RAG → final answer
    tools/
      mandi.py              # data.gov.in (Agmarknet) client + caching
      pricing.py            # Quantile LightGBM (p20/p50/p80) + SELL/WAIT
      weather.py            # 24h + 7d forecast summary
      sentinel.py           # Sentinel Hub indices (NDVI/NDMI/NDWI/LAI)
      lang.py               # Language detection and translation
      geocode.py            # Location name → Lat/Lon
      geo.py                # Lat/Lon → State/District
    rag/
      index.py              # Chroma index build/load
      retrieve.py           # Top-k passages
      generate.py           # LLM synthesis (tool+RAG aware)
    utils/cache.py          # Simple in-memory TTL cache
bots/
  telegram/bot.py           # Chat, slot filling, voice, disease photos
vit_disease/
  vit_model.py              # Local ViT classifier (leaf disease)
  llm_helper.py             # Care advice text via LLM
models/
  pricing_global/           # p20/p50/p80 models + meta + encoder
```

---

## 🔑 Environment Variables (.env)

Use `.env.example` as a template. You will need to provide the following keys:

```env
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
```

---

## 🧪 How to Test Locally

### Backend API

**Health Check:**

```bash
curl -s http://127.0.0.1:8000/health
```

**Full-Featured Query:**

```bash
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
```

### Telegram Bot

Start a chat with your bot and try these commands:

- Share your location (📍)
- Set market: Karnataka, Bangalore, Ramanagara
- Set crop tomato
- Ask a question: "Should I sell now?" or "कल बारिश होगी?"
- Send a photo of a diseased plant leaf
- Send a voice note with your question
