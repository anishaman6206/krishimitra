# KrishiMitra AI — The Farmer's Copilot 🇮🇳

A unified, data-driven assistant for Indian farmers, designed to provide actionable, real-time advice on the most critical aspects of agriculture.

This project brings together a suite of powerful tools to answer a farmer's most pressing questions, from market prices to crop health, all through a simple, accessible interface.

---

## ✨ Core Features

- **📈 Market Price Forecasting:** Fetches live prices from Agmarknet (data.gov.in) and uses ML to provide a farmer-friendly Sell / Wait recommendation for the next 1-2 weeks.
- **🛰️ Real-Time Satellite Analysis:** Leverages Sentinel-2 data to analyze vegetation health (NDVI, NDMI, NDWI, LAI), automatically adjusting the area of interest to find cloud-free images.
- **🌦️ Hyperlocal Weather Forecasts:** Provides detailed 7-day and 24-hour weather summaries for the farmer's specific location.
- **📚 Fact-Grounded RAG:** Uses a Retrieval-Augmented Generation pipeline over a curated **Agri Knowledge Base** (seeded with official government documents and agricultural university guidelines) to answer complex queries without hallucination.
- **🌍 Location Intelligence:**
  - `geocode.py`: Converts location names (e.g., "Kharagpur") into precise latitude and longitude using OpenStreetMap.
  - `geo.py`: Performs reverse geocoding to identify the state, district, etc., from geographic coordinates.
- **🗣️ Multilingual Support:**
  - `lang.py`: Automatically detects the user's language and provides answers in the same language for a natural, intuitive experience.
- **🌿 Leaf Disease Detection:** A local Vision Transformer (ViT) model identifies common plant diseases from a photo, with the LLM providing tailored care advice.
- **🤖 Accessible Bots:** Deployed via a Telegram bot with a user-friendly interface, including voice commands (STT/TTS) and quick-action buttons.

---

### 📸 Key Features in Action
Here are some screenshots showcasing the KrishiMitra AI bot in action.

- Welcome & Location. Onboarding new users with multilingual examples.
<img width="630" height="460" alt="Screenshot 2025-08-19 105251" src="https://github.com/user-attachments/assets/38b88e56-d17c-48f1-be37-00b8a42f3d99" />
<img width="489" height="519" alt="image" src="https://github.com/user-attachments/assets/7a959e34-8d0e-4b90-8702-e8a9847ce1fd" />
<img width="635" height="697" alt="Screenshot 2025-08-19 105346" src="https://github.com/user-attachments/assets/dfa1747f-3ea4-491e-9022-2016b1c5731c" />
<img width="615" height="849" alt="Screenshot 2025-08-19 105358" src="https://github.com/user-attachments/assets/30b375f5-15ee-44b4-9bfe-76b281c46d3f" />
<img width="626" height="867" alt="Screenshot 2025-08-19 105412" src="https://github.com/user-attachments/assets/7e24fb92-d546-4355-8ac1-71245cc04679" />
<img width="609" height="758" alt="Screenshot 2025-08-19 105429" src="https://github.com/user-attachments/assets/590b3c5b-1aa6-4a15-9247-c977abb9afe0" />
<img width="609" height="331" alt="Screenshot 2025-08-19 105447" src="https://github.com/user-attachments/assets/a91860a6-e680-4bee-9e8c-9536b0333ff2" />
<img width="363" height="668" alt="image" src="https://github.com/user-attachments/assets/c21ad1be-63ff-4374-9e2e-59700c683379" />

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

### 🌊 Architecture Flow Diagram

<img width="3840" height="2816" alt="Mermaid Chart - Create complex, visual diagrams with text  A smarter way of creating diagrams -2025-08-19-055222" src="https://github.com/user-attachments/assets/043b903e-f94b-4131-911e-dde1b1f33b65" />

### Telegram Bot

Start a chat with your bot and try these commands:

- Share your location (📍)
- Ask a question:
- "Should I sell now?"
- "कल बारिश होगी?"
- "What seed variety suits this unpredictable weather?"
- "क्या मुझे इस हफ्ते टमाटर बेचना चाहिए?"
- "Will next week’s temperature drop kill my yield?"
- "Kal ka mausam kaisa rahega?"

- Send a photo of a diseased plant leaf.
- Send a voice note with your question.
