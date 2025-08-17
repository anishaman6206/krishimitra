# bots/telegram/bot.py
import os
import json
import logging
import hashlib
from typing import Any, Dict, Optional, Tuple, List

import httpx
from dotenv import load_dotenv

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InputFile,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# --------------------
# ENV & CONFIG
# --------------------
load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
# Accept either BACKEND_URL or BACKEND_BASE_URL
BACKEND_URL = os.getenv("BACKEND_URL") or os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")
BOT_DEFAULT_CROP = os.getenv("BOT_DEFAULT_CROP", "Tomato")
URL_ASK = f"{BACKEND_URL.rstrip('/')}/ask"
URL_HEALTH = f"{BACKEND_URL.rstrip('/')}/health"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("krishimitra.bot")

# --------------------
# Commodity detection
# --------------------
_COMMODITY_ALIASES = {
    "tomato": "Tomato", "टमाटर": "Tomato",
    "onion": "Onion", "प्याज": "Onion",
    "potato": "Potato", "आलू": "Potato",
    "chilli": "Chilli", "chili": "Chilli", "mirchi": "Chilli", "मिर्च": "Chilli",
    "banana": "Banana", "केला": "Banana",
    "brinjal": "Brinjal", "eggplant": "Brinjal", "बैंगन": "Brinjal",
    "okra": "Okra", "bhindi": "Okra", "भिंडी": "Okra",
    "cabbage": "Cabbage", "पत्ता गोभी": "Cabbage", "गोभी": "Cabbage",
    "cauliflower": "Cauliflower", "फूलगोभी": "Cauliflower",
    "wheat": "Wheat", "गेहूं": "Wheat", "गेहू": "Wheat",
    "paddy": "Paddy", "rice": "Paddy", "धान": "Paddy",
    "maize": "Maize", "corn": "Maize", "मक्का": "Maize",
    "gram": "Gram", "chana": "Gram", "चना": "Gram",
    "tur": "Arhar", "arhar": "Arhar", "pigeon pea": "Arhar", "अरहर": "Arhar", "तूर": "Arhar",
    "moong": "Moong", "green gram": "Moong", "मूंग": "Moong",
    "urad": "Urad", "black gram": "Urad", "उड़द": "Urad",
    "soybean": "Soyabean", "soya": "Soyabean", "सोयाबीन": "Soyabean",
    "groundnut": "Groundnut", "peanut": "Groundnut", "मूंगफली": "Groundnut",
    "mustard": "Mustard", "सरसों": "Mustard",
    "cotton": "Cotton", "कपास": "Cotton",
    "sugarcane": "Sugarcane", "गन्ना": "Sugarcane",
}

MARKET_KEYWORDS = (
    "sell", "wait", "hold", "price", "market", "mandi", "rate", "bhav", "bech",
    "kab bechna", "kab bechu", "kab bechen", "kinna milega", "kitna bhav"
)

def _extract_crop(text: str) -> Optional[str]:
    if not text: return None
    t = text.lower()
    words = [w.strip(".,?!:;()[]{}\"'") for w in t.split()]
    for w in words:
        if w == "price":  # avoid 'price' → 'rice'
            continue
        if w in _COMMODITY_ALIASES:
            return _COMMODITY_ALIASES[w]
    for k, v in _COMMODITY_ALIASES.items():
        if " " in k and k in t:
            return v
    return None

# --------------------
# Intent & slots
# --------------------
INTENT_KEYWORDS = {
    "SELL":    ["sell", "wait", "mandi", "price", "बेचना", "रुको", "कीमत", "mandya", "market"],
    "WEATHER": ["weather", "rain", "forecast", "barish", "rainfall", "मौसम", "बारिश"],
    "NDVI":    ["ndvi", "satellite", "crop health", "field image", "सैटेलाइट"],
    "IRRIG":   ["irrigate", "irrigation", "water now", "सिंचाई", "पानी"],
    "POLICY":  ["scheme", "policy", "credit", "loan", "pmfby", "insurance", "योजना", "कर्ज"],
}

REQUIRED_SLOTS = {
    "SELL":   ["crop"],             # location helps but pricing can still work best-effort
    "WEATHER":["geo"],
    "NDVI":   ["geo"],
    "IRRIG":  ["crop", "geo"],
    "POLICY": [],
    "GENERIC":[]
}

def _detect_intent(text: str) -> str:
    t = (text or "").lower()
    for intent, kws in INTENT_KEYWORDS.items():
        if any(k in t for k in kws):
            return intent
    return "GENERIC"

# --------------------
# Per-user prefs in user_data
# --------------------
def _prefs(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> Dict[str, Any]:
    ud = context.user_data
    if "prefs" not in ud:
        ud["prefs"] = {
            "lat": None, "lon": None,
            "last_tool_notes": {}, "last_sources": [],
            "last_crop": None, "last_state": None, "last_district": None, "last_market": None,
            "horizon_days": 7,
            "lang": None,
            "dialog": {"intent": None, "slots": {}, "pending": None},  # slot-filling state
        }
    return ud["prefs"]

def _split_chunks(s: str, n: int = 4000) -> List[str]:
    return [s[i:i+n] for i in range(0, len(s), n)]

def _mk_reply_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [[KeyboardButton("📍 Share location", request_location=True)],
         [KeyboardButton("🧾 Example: Should I sell tomatoes now?")]],
        resize_keyboard=True
    )

def _is_market_query(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in MARKET_KEYWORDS)

def _mk_inline_kb(tool_notes: dict | None, user_text: str | None = None) -> InlineKeyboardMarkup | None:
    rows = []

    # # NDVI quicklook only if available from last run
    # ql = (tool_notes or {}).get("ndvi_quicklook")
    # if isinstance(ql, dict) and ql.get("available"):
    #     rows.append([InlineKeyboardButton("🛰️ Open NDVI image", callback_data="NDVI")])

    # Price horizon buttons ONLY for market queries
    if _is_market_query(user_text or ""):
        rows.append([
            InlineKeyboardButton("Set horizon: 7d", callback_data="h7"),
            InlineKeyboardButton("Set horizon: 14d", callback_data="h14"),
        ])
        rows.append([InlineKeyboardButton("⚙️ Set market", callback_data="set_market")])

    return InlineKeyboardMarkup(rows) if rows else None

async def _send_action(context: ContextTypes.DEFAULT_TYPE, chat_id: int, action: ChatAction):
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=action)
    except Exception:
        pass

def _ensure_backend_url(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    return f"{BACKEND_URL.rstrip('/')}/{path_or_url.lstrip('/')}"

async def _backend_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(URL_HEALTH)
            return r.status_code == 200
    except Exception:
        return False

async def _call_ask(payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(URL_ASK, json=payload)
        r.raise_for_status()
        return r.json()

def _guess_lang(update: Update) -> Optional[str]:
    try: return update.effective_user.language_code
    except Exception: return None

# --------------------
# Reverse geocoding (optional but helpful)
# --------------------
async def _reverse_geocode(lat: float, lon: float) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (state, district) via Nominatim (no key, rate-limited).
    """
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"format": "jsonv2", "lat": str(lat), "lon": str(lon), "zoom": "10", "addressdetails": 1}
    headers = {"User-Agent": "KrishiMitra/1.0"}
    try:
        async with httpx.AsyncClient(timeout=12, headers=headers) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            js = r.json()
        addr = js.get("address", {})
        state = addr.get("state")
        # district may appear as 'district', 'county', or 'state_district'
        district = addr.get("district") or addr.get("county") or addr.get("state_district")
        return state, district
    except Exception:
        return None, None

# --------------------
# Slot helpers
# --------------------
def _get_missing_slots(intent: str, prefs: Dict[str, Any], parsed_crop: Optional[str]) -> List[str]:
    needed = REQUIRED_SLOTS.get(intent, [])
    missing = []
    if "crop" in needed:
        crop = parsed_crop or prefs.get("last_crop")
        if not crop: missing.append("crop")
    if "geo" in needed:
        if prefs.get("lat") is None or prefs.get("lon") is None:
            missing.append("geo")
    return missing

async def _ask_for_slot(update: Update, context: ContextTypes.DEFAULT_TYPE, slot: str):
    if slot == "geo":
        await update.message.reply_text("📍 Please share your location to personalize weather & NDVI.",
                                        reply_markup=_mk_reply_kb())
    elif slot == "crop":
        await update.message.reply_text("🌾 Which crop are you asking about? (e.g., Tomato, Onion, Wheat)")
    else:
        await update.message.reply_text(f"Please provide: {slot}")

async def _proceed_with_payload(update: Update, context: ContextTypes.DEFAULT_TYPE,
                                intent: str, crop: Optional[str]):
    chat_id = update.effective_chat.id
    prefs = _prefs(context, chat_id)

    # Resolve state/district from lat/lon if missing
    if prefs.get("lat") is not None and prefs.get("lon") is not None:
        if prefs.get("last_state") is None or prefs.get("last_district") is None:
            st, dist = await _reverse_geocode(prefs["lat"], prefs["lon"])
            if st:   prefs["last_state"] = st
            if dist: prefs["last_district"] = dist

    await _send_action(context, chat_id, ChatAction.TYPING)

    # Enable pricing if: explicit SELL intent OR text looks market-related
    text_now = update.message.text or ""
    marketish = (intent == "SELL") or _is_market_query(text_now)

    crop_for_payload = (crop or prefs.get("last_crop") or BOT_DEFAULT_CROP) if marketish else None

    payload = {
        "text": text_now,
        "lang": prefs.get("lang"),
        "crop": crop_for_payload,                     # ← only when marketish
        "state": prefs.get("last_state"),
        "district": prefs.get("last_district"),
        "market": prefs.get("last_market"),
        "horizon_days": prefs.get("horizon_days", 7),
        "geo": ({"lat": prefs["lat"], "lon": prefs["lon"]}
                if (prefs.get("lat") is not None and prefs.get("lon") is not None) else None),
        "debug": False,
    }

    try:
        data = await _call_ask(payload)
    except httpx.HTTPError as e:
        await update.message.reply_text(
            f"⚠️ Backend error: {e}", reply_markup=_mk_reply_kb()
        )
        return

    tool_notes = data.get("tool_notes") or {}
    prefs["last_tool_notes"] = tool_notes
    # keep sources around in case you later expose a Sources button
    #prefs["last_sources"] = data.get("sources") or []

    # Remember resolved commodity/location if pricing ran
    price_ctx = tool_notes.get("price") or tool_notes.get("sell_wait", {}).get("context")
    if isinstance(price_ctx, dict):
        prefs["last_crop"] = price_ctx.get("commodity") or crop or prefs.get("last_crop")
        if price_ctx.get("state"):    prefs["last_state"] = price_ctx["state"]
        if price_ctx.get("district"): prefs["last_district"] = price_ctx["district"]
        if price_ctx.get("market"):   prefs["last_market"] = price_ctx["market"]

    # Final answer
    answer = data.get("answer") or "Sorry, I couldn't generate an answer."
    for chunk in _split_chunks(answer):
        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)

    kb = _mk_inline_kb(tool_notes, user_text=update.message.text)
    if kb:
        await update.message.reply_text("—", reply_markup=kb)

    # reset dialog
    prefs["dialog"] = {"intent": None, "slots": {}, "pending": None}

# --------------------
# Voice helpers (uses voice_utils.py)
# --------------------
async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Accept a Telegram voice note, run STT, call backend with detected language, and offer a TTS reply."""
    try:
        # show typing
        await _send_action(context, update.effective_chat.id, ChatAction.TYPING)

        from voice_utils import opus_to_wav, wav_to_opus, sarvam_stt, text_to_speech

        chat_id = update.effective_chat.id
        prefs = _prefs(context, chat_id)

        file_id = update.message.voice.file_id
        t_short = hashlib.md5(file_id.encode()).hexdigest()[:12]
        ogg = f"voice_{chat_id}_{t_short}.ogg"
        wav = f"voice_{chat_id}_{t_short}.wav"

        file = await context.bot.get_file(file_id)
        await file.download_to_drive(ogg)
        await opus_to_wav(ogg, wav)

        transcript, detected_lang = await sarvam_stt(wav)
        if not transcript.strip():
            await update.message.reply_text("Sorry, I couldn't understand the audio.")
            return

        # call backend with detected language
        payload = {
            "text": transcript,
            "lang": detected_lang,
            "crop": prefs.get("last_crop") if _is_market_query(transcript) else None,
            "state": prefs.get("last_state"),
            "district": prefs.get("last_district"),
            "market": prefs.get("last_market"),
            "horizon_days": prefs.get("horizon_days", 7),
            "geo": ({"lat": prefs["lat"], "lon": prefs["lon"]}
                    if (prefs.get("lat") is not None and prefs.get("lon") is not None) else None),
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(URL_ASK, json=payload)
            r.raise_for_status()
            data = r.json()

        prefs["last_tool_notes"] = data.get("tool_notes") or {}
        answer_text = data.get("answer") or "I couldn't form an answer."

        # present a button to hear the answer
        kb = InlineKeyboardMarkup([[InlineKeyboardButton("🔊 Hear this answer", callback_data=f"hear_{t_short}")]])
        await update.message.reply_text(f"📝 {answer_text}", reply_markup=kb)

        # store answer for TTS callback
        context.user_data[f"hear_{t_short}"] = {
            "text": answer_text,
            "language": detected_lang,
            "chat_id": chat_id
        }

        # cleanup local files
        for f in (ogg, wav):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass

    except Exception as e:
        logging.exception("Voice processing error", exc_info=e)
        await update.message.reply_text(f"Sorry, there was an error processing your voice message: {e}")

async def on_hear_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate and send TTS when user taps 'Hear this answer'."""
    query = update.callback_query
    await query.answer()
    if not query.data.startswith("hear_"):
        return

    short_id = query.data[5:]
    stash = context.user_data.get(f"hear_{short_id}")
    if not stash:
        await query.edit_message_text("This answer has expired. Please ask again.")
        return

    text = stash["text"]
    language = stash["language"]
    chat_id = stash["chat_id"]

    try:
        from voice_utils import text_to_speech, wav_to_opus
        await query.edit_message_text(f"🔊 Generating voice reply...\n\n📝 {text}")

        wav = f"reply_{chat_id}_{short_id}.wav"
        ogg = f"reply_{chat_id}_{short_id}.ogg"
        tts_lang = f"{language}-IN" if language and language != "en" else "en-IN"

        await text_to_speech(text, wav, "default", tts_lang)
        await wav_to_opus(wav, ogg)

        with open(ogg, "rb") as vf:
            await context.bot.send_voice(chat_id, voice=vf)

        await query.edit_message_reply_markup(
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("✅ Voice sent!", callback_data="voice_sent")]])
        )

    except Exception as e:
        await query.edit_message_text(f"Sorry, there was an error generating voice: {e}")
    finally:
        # cleanup
        for f in (wav, ogg):
            try:
                if os.path.exists(f):
                    os.remove(f)
            except Exception:
                pass
        try:
            del context.user_data[f"hear_{short_id}"]
        except Exception:
            pass

# --------------------
# Command / message handlers
# --------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prefs = _prefs(context, update.effective_chat.id)
    prefs["lang"] = prefs.get("lang") or _guess_lang(update)
    await update.message.reply_text(
        "👋 नमस्ते / Hello! I’m **KrishiMitra AI**.\n\n"
        "Ask: *“Should I sell tomatoes now?”*, *“When should I irrigate?”*, *“Will it rain tomorrow?”*.\n"
        "Tip: Share your location for local weather & NDVI.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=_mk_reply_kb()
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "You can ask:\n"
        "• When should I irrigate?\n"
        "• Should I sell tomatoes now?\n"
        "• Will next week’s temperature drop hurt my crop?\n"
        "• What schemes can help with finances?\n\n"
        "Share your location to enable local weather & NDVI.",
        reply_markup=_mk_reply_kb()
    )

async def health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok = await _backend_health()
    await update.message.reply_text("✅ Backend OK" if ok else f"❌ Backend unhealthy at {BACKEND_URL}")

async def on_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    loc = update.message.location
    prefs = _prefs(context, update.effective_chat.id)
    prefs["lat"] = loc.latitude
    prefs["lon"] = loc.longitude
    await update.message.reply_text(f"✅ Location saved: {loc.latitude:.5f}, {loc.longitude:.5f}\nNow ask your question.",
                                    reply_markup=_mk_reply_kb())

    # auto-continue if we were waiting for geo
    dlg = prefs.get("dialog") or {}
    if dlg.get("pending") == "geo":
        intent = dlg.get("intent") or "GENERIC"
        crop = dlg.get("slots", {}).get("crop") or prefs.get("last_crop")
        await _proceed_with_payload(update, context, intent, crop)

async def on_example(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await on_text(update, context, forced_text="Should I sell tomatoes now?")

def _get_missing_slots_for_intent(text: str, prefs: Dict[str, Any], parsed_crop: Optional[str]) -> List[str]:
    intent = _detect_intent(text)
    return _get_missing_slots(intent, prefs, parsed_crop)

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE, forced_text: Optional[str] = None):
    msg = forced_text or (update.message.text or "").strip()
    if not msg: return

    chat_id = update.effective_chat.id
    prefs = _prefs(context, chat_id)
    prefs["lang"] = prefs.get("lang") or _guess_lang(update)

    # Quick "Set market:" parser (free-form text)
    if msg.lower().startswith("set market:"):
        try:
            payload = msg.split(":", 1)[1].strip()
            parts = [p.strip() for p in payload.split(",")]
            if len(parts) >= 1: prefs["last_state"] = parts[0] or None
            if len(parts) >= 2: prefs["last_district"] = parts[1] or None
            if len(parts) >= 3: prefs["last_market"] = parts[2] or None
            await update.message.reply_text(
                f"✅ Set market:\nState={prefs['last_state']}\nDistrict={prefs['last_district']}\nMarket={prefs['last_market']}"
            )
        except Exception:
            await update.message.reply_text("Format: Set market: Karnataka, Bangalore, Ramanagara")
        return

    # Allow free “Set crop … / Set horizon …” text
    if msg.lower().startswith("set crop"):
        prefs["last_crop"] = _extract_crop(msg) or msg.split(" ", 2)[-1].strip().title()
        await update.message.reply_text(f"✅ Crop set to: {prefs['last_crop']}")
        return
    if msg.lower().startswith("set horizon"):
        try:
            prefs["horizon_days"] = int(msg.split()[-1])
            await update.message.reply_text(f"✅ Horizon set to: {prefs['horizon_days']} days")
        except Exception:
            await update.message.reply_text("Say: Set horizon 7  (or 14)")
        return

    # If we were filling a pending slot, try to fill it now
    dlg = prefs.get("dialog") or {"intent": None, "slots": {}, "pending": None}
    if dlg.get("pending"):
        need = dlg["pending"]
        if need == "crop":
            crop = _extract_crop(msg) or msg.title().strip()
            dlg["slots"]["crop"] = crop
            dlg["pending"] = None
            prefs["dialog"] = dlg
            await _proceed_with_payload(update, context, dlg.get("intent") or "GENERIC", crop)
            return
        # other slots can be added similarly

    # New turn: detect intent & parse quick slots
    intent = _detect_intent(msg)
    parsed_crop = _extract_crop(msg)
    missing = _get_missing_slots(intent, prefs, parsed_crop)

    if missing:
        # ask first missing, store dialog state
        first = missing[0]
        prefs["dialog"] = {"intent": intent, "slots": {}, "pending": first}
        if parsed_crop and "crop" not in missing:
            prefs["dialog"]["slots"]["crop"] = parsed_crop
        await _ask_for_slot(update, context, first)
        return

    # All good — proceed straight to backend
    await _proceed_with_payload(update, context, intent, parsed_crop)

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    prefs = _prefs(context, update.effective_chat.id)
    tool_notes = prefs.get("last_tool_notes") or {}
    data = query.data or ""

    if data == "NDVI":
        ql = (tool_notes or {}).get("ndvi_quicklook")
        if not (isinstance(ql, dict) and ql.get("available")):
            await query.message.reply_text("No NDVI image available from the last answer.")
            return
        url = _ensure_backend_url(ql.get("url"))
        await _send_action(context, update.effective_chat.id, ChatAction.UPLOAD_PHOTO)
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(url)
                r.raise_for_status()
                content = r.content
        except Exception as e:
            await query.message.reply_text(f"Couldn't fetch NDVI image: {e}")
            return
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=InputFile.from_bytes(content, filename="ndvi.png"),
            caption="🛰️ NDVI quicklook"
        )
        return

    if data == "h7":
        prefs["horizon_days"] = 7
        await query.message.reply_text("✅ Horizon set to 7 days.")
        return
    if data == "h14":
        prefs["horizon_days"] = 14
        await query.message.reply_text("✅ Horizon set to 14 days.")
        return
    if data == "set_market":
        await query.message.reply_text("Send me your market like:\nSet market: Karnataka, Bangalore, Ramanagara")
        return

async def on_hear_sent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Acknowledge "voice_sent" callback
    await update.callback_query.answer("Voice message sent! 🎉")

# --------------------
# Main
# --------------------
def main():
    if not BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set in environment")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("health", health))

    app.add_handler(MessageHandler(filters.LOCATION, on_location))
    app.add_handler(MessageHandler(filters.Regex("^🧾 Example:"), on_example))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))

    # Voice callbacks
    app.add_handler(CallbackQueryHandler(on_hear_callback, pattern="^hear_"))
    app.add_handler(CallbackQueryHandler(on_hear_sent, pattern="^voice_sent$"))

    # Generic button callbacks (NDVI, horizons, set_market)
    app.add_handler(CallbackQueryHandler(on_callback))

    log.info("KrishiMitra Telegram bot (slot-filling + voice) started.")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
