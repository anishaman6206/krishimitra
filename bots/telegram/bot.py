import os
import json
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from pathlib import Path
import httpx
# Add these imports at the top of the file
import sys
sys.path.append(str(Path(__file__).parents[2] / "backend"))
from app.tools.lang import translate_to_en, translate_from_en

from telegram import (
    Update,
    KeyboardButton,
    ReplyKeyboardMarkup,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InputFile,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from telegram.constants import ChatAction
import logging

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logging.exception("Telegram error", exc_info=context.error)

# ----- env -----
dotenv_path = Path(__file__).parents[2] / ".env"
load_dotenv(dotenv_path)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BACKEND_BASE = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")
ASK_URL = f"{BACKEND_BASE.rstrip('/')}/ask"

if not BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")

# ----- tiny per-chat state -----
@dataclass
class ChatPrefs:
    crop: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    market: Optional[str] = None
    horizon_days: int = 7
    lat: Optional[float] = None
    lon: Optional[float] = None
    last_tool_notes: Dict[str, Any] = field(default_factory=dict)

PREFS: Dict[int, ChatPrefs] = {}

# ----- helpers -----
def _prefs(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int) -> ChatPrefs:
    if chat_id not in PREFS:
        PREFS[chat_id] = ChatPrefs()
    return PREFS[chat_id]

def _reply_kb():
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton("📍 Share location", request_location=True)],
            [KeyboardButton("Set crop"), KeyboardButton("Set horizon")],
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
    )

async def _ask_backend(prompt: str, prefs: ChatPrefs, detected_lang: str = "en", debug: bool = False) -> Dict[str, Any]:
    payload = {
        "text": prompt,
        "lang": detected_lang,  # Pass detected language to backend
        "crop": prefs.crop,
        "state": prefs.state,
        "district": prefs.district,
        "market": prefs.market,
        "horizon_days": prefs.horizon_days,
        "geo": {"lat": prefs.lat, "lon": prefs.lon} if (prefs.lat and prefs.lon) else None,
        "debug": debug,
    }
    # strip None
    payload = {k: v for k, v in payload.items() if v is not None}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(ASK_URL, json=payload)
        r.raise_for_status()
        return r.json()
    

def _followup_buttons(resp: Dict[str, Any]) -> InlineKeyboardMarkup | None:
    btns = []
    for item in (resp.get("follow_ups") or []):
        label = str(item)
        if label.lower().startswith("open ndvi image"):
            btns.append([InlineKeyboardButton("🛰️ Open NDVI image", callback_data="ndvi_img")])
        elif "week 1" in label.lower() or "week 2" in label.lower():
            # optional: ignore, your answer already prints forecast
            continue
        elif "compare another market" in label.lower():
            btns.append([InlineKeyboardButton("🧭 Set market", callback_data="set_market")])
        elif "change holding period" in label.lower():
            btns.append([
                InlineKeyboardButton("Set horizon: 7d", callback_data="hz_7"),
                InlineKeyboardButton("Set horizon: 14d", callback_data="hz_14"),
            ])
        elif "ask this in my language" in label.lower():
            # can map to the user's Telegram language later if you want
            continue
    return InlineKeyboardMarkup(btns) if btns else None

def _static_abs(url_path: str) -> str:
    # make /static/... into absolute http URL
    if url_path.startswith("http://") or url_path.startswith("https://"):
        return url_path
    return f"{BACKEND_BASE.rstrip('/')}{url_path}"

# ----- handlers -----
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    _prefs(context, chat_id)  # ensure exists
    msg = (
        "👋 Namaste! I’m KrishiMitra.\n"
        "Send me a question like:\n"
        "• *Can I wait to sell tomatoes?*\n"
        "• *Will next week’s temperature hurt my crop?*\n\n"
        "Tip: tap ‘📍 Share location’ so I can use local weather & NDVI."
    )
    await update.message.reply_text(msg, reply_markup=_reply_kb(), parse_mode="Markdown")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n"
        "/setcrop Tomato | Wheat | Paddy\n"
        "/sethorizon 7 or 14\n"
        "Or just ask a question in plain language.\n"
        "Use the keyboard to share your location.",
        reply_markup=_reply_kb()
    )

async def setcrop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    prefs = _prefs(context, chat_id)
    args = (context.args or [])
    if not args:
        await update.message.reply_text("Usage: /setcrop Tomato")
        return
    prefs.crop = " ".join(args).strip()
    await update.message.reply_text(f"✅ Crop set to: {prefs.crop}")

async def sethorizon_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    prefs = _prefs(context, chat_id)
    args = (context.args or [])
    if not args or not args[0].isdigit() or int(args[0]) not in (7, 14):
        await update.message.reply_text("Usage: /sethorizon 7  (or 14)")
        return
    prefs.horizon_days = int(args[0])
    await update.message.reply_text(f"✅ Horizon set to: {prefs.horizon_days} days")

async def on_location(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    prefs = _prefs(context, chat_id)
    loc = update.message.location
    if not loc:
        return
    prefs.lat = float(loc.latitude)
    prefs.lon = float(loc.longitude)
    await update.message.reply_text(
        f"📍 Got your location: {prefs.lat:.5f}, {prefs.lon:.5f}\n"
        "Ask about prices, weather, or NDVI now."
    )

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    prefs = _prefs(context, chat_id)
    text = (update.message.text or "").strip()

    # quick UI actions without slash
    if text.lower().startswith("set crop"):
        prefs.crop = text.split(" ", 2)[-1].strip()
        await update.message.reply_text(f"✅ Crop set to: {prefs.crop}")
        return
    if text.lower().startswith("set horizon"):
        try:
            prefs.horizon_days = int(text.split()[-1])
            await update.message.reply_text(f"✅ Horizon set to: {prefs.horizon_days} days")
        except Exception:
            await update.message.reply_text("Say: Set horizon 7  (or 14)")
        return

    # little UX hint: show “typing…”
    try:
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        # For text messages, assume English unless you implement language detection
        resp = await _ask_backend(text, prefs, "en")  # Add language parameter
        prefs.last_tool_notes = resp.get("tool_notes") or {}
        answer = resp.get("answer") or "I couldn't form an answer."
        kb = _followup_buttons(resp)

        await update.message.reply_text(answer, reply_markup=kb, disable_web_page_preview=True)
    except httpx.HTTPError as e:
        await update.message.reply_text(f"Backend error: {e}")
    except Exception as e:
        await update.message.reply_text(f"Oops: {e}")

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = update.effective_chat.id
    prefs = _prefs(context, chat_id)

    data = query.data or ""
    if data == "ndvi_img":
        notes = prefs.last_tool_notes or {}
        ql = (notes.get("ndvi_quicklook") or {})
        url_path = ql.get("url")
        if not url_path:
            await query.edit_message_text("No NDVI image available for this query.")
            return
        abs_url = _static_abs(url_path)
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(abs_url)
                r.raise_for_status()
                b = r.content
            await context.bot.send_photo(chat_id=chat_id, photo=b, caption="NDVI quicklook")
        except Exception as e:
            await context.bot.send_message(chat_id, f"Couldn't fetch NDVI image: {e}")

    elif data == "hz_7":
        prefs.horizon_days = 7
        await context.bot.send_message(chat_id, "✅ Horizon set to 7 days.")
    elif data == "hz_14":
        prefs.horizon_days = 14
        await context.bot.send_message(chat_id, "✅ Horizon set to 14 days.")
    elif data == "set_market":
        await context.bot.send_message(
            chat_id,
            "Send me your market like:\nSet market: Karnataka, Bangalore, Ramanagara"
        )
    else:
        await context.bot.send_message(chat_id, "Action not implemented yet.")

async def on_text_market(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Quick parser for: Set market: State, District, Market"""
    chat_id = update.effective_chat.id
    prefs = _prefs(context, chat_id)
    text = (update.message.text or "").strip()
    if not text.lower().startswith("set market"):
        return
    try:
        payload = text.split(":", 1)[1].strip()
        parts = [p.strip() for p in payload.split(",")]
        if len(parts) >= 1: prefs.state = parts[0] or None
        if len(parts) >= 2: prefs.district = parts[1] or None
        if len(parts) >= 3: prefs.market = parts[2] or None
        await update.message.reply_text(
            f"✅ Set market:\nState={prefs.state}\nDistrict={prefs.district}\nMarket={prefs.market}"
        )
    except Exception:
        await update.message.reply_text(
            "Format: Set market: Karnataka, Bangalore, Ramanagara"
        )



async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from voice_utils import opus_to_wav, wav_to_opus, sarvam_stt, text_to_speech
    import hashlib
    
    chat_id = update.effective_chat.id
    prefs = _prefs(context, chat_id)
    
    try:
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        
        # 1-3. Download, convert, and transcribe (same as before)
        file_id = update.message.voice.file_id
        file = await context.bot.get_file(file_id)
        short_id = hashlib.md5(file_id.encode()).hexdigest()[:12]
        
        input_ogg = f"voice_{chat_id}_{short_id}.ogg"
        await file.download_to_drive(input_ogg)
        
        input_wav = f"voice_{chat_id}_{short_id}.wav"
        await opus_to_wav(input_ogg, input_wav)
        
        transcript, detected_lang = await sarvam_stt(input_wav)
        
        if not transcript.strip():
            await update.message.reply_text("Sorry, I couldn't understand the audio.")
            return
        
        print(f"STT Result: '{transcript}' (Language: {detected_lang})")
        
        # 4. Send original transcript with detected language to backend
        # Backend will handle translation internally
        resp = await _ask_backend(transcript, prefs, detected_lang)
        answer_local = resp.get("answer", "I couldn't form an answer.")  # Backend returns translated answer
        print(f"Backend response: {resp}")

        # 5. Send text reply (backend already translated if needed)
        voice_keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("🔊 Hear this answer", callback_data=f"hear_{short_id}")
        ]])
        
        await update.message.reply_text(
            f"📝 {answer_local}", 
            reply_markup=voice_keyboard
        )
        
        # 6. Store answer for voice generation
        context.user_data[f"hear_{short_id}"] = {
            "text": answer_local,
            "language": detected_lang,
            "chat_id": chat_id
        }
        
        # 7. Clean up input files
        import os
        for f in [input_ogg, input_wav]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass
                
    except Exception as e:
        print(f"Voice processing error: {e}")
        await update.message.reply_text(f"Sorry, there was an error processing your voice message: {e}")


# Update handle_voice_request to use short_id:
async def handle_voice_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle when user clicks 'Hear this answer' button"""
    query = update.callback_query
    await query.answer()
    
    # Extract short_id from callback data
    if not query.data.startswith("hear_"):
        return
    
    short_id = query.data[5:]  # Remove "hear_" prefix
    answer_data = context.user_data.get(f"hear_{short_id}")
    
    if not answer_data:
        await query.edit_message_text("Sorry, this answer has expired. Please ask your question again.")
        return
    
    answer_text = answer_data["text"]
    language = answer_data["language"]
    chat_id = answer_data["chat_id"]
    
    try:
        from voice_utils import text_to_speech, wav_to_opus
        
        # Show that we're generating voice
        await query.edit_message_text(f"🔊 Generating voice reply...\n\n📝 {answer_text}")
        
        # Generate voice reply
        reply_wav = f"reply_{chat_id}_{short_id}.wav"
        
        # Pass language to TTS for better pronunciation
        tts_lang = f"{language}-IN" if language != "en" else "en-IN"
        await text_to_speech(answer_text, reply_wav, "default", tts_lang)
        
        # Convert to OGG/Opus for Telegram
        reply_ogg = f"reply_{chat_id}_{short_id}.ogg"
        await wav_to_opus(reply_wav, reply_ogg)
        
        # Send voice message
        with open(reply_ogg, "rb") as voice_file:
            await context.bot.send_voice(chat_id, voice=voice_file)
        
        # Update the button to show it's been sent
        sent_keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Voice sent!", callback_data="voice_sent")
        ]])
        await query.edit_message_reply_markup(reply_markup=sent_keyboard)
        
        # Clean up files
        import os
        for f in [reply_wav, reply_ogg]:
            try:
                if os.path.exists(f):
                    os.remove(f)
            except:
                pass
        
        # Clean up stored data
        del context.user_data[f"hear_{short_id}"]
        
    except Exception as e:
        await query.edit_message_text(f"Sorry, there was an error generating voice: {e}")

# Add a handler for the "Voice sent!" button (just to acknowledge)
async def handle_voice_sent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer("Voice message sent! 🎉")

# Update the main() function to include the new handlers:
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("setcrop", setcrop_cmd))
    app.add_handler(CommandHandler("sethorizon", sethorizon_cmd))

    app.add_handler(MessageHandler(filters.LOCATION, on_location))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r"^Set market:"), on_text_market))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.VOICE, on_voice))

    # Add new callback handlers for voice functionality
    app.add_handler(CallbackQueryHandler(handle_voice_request, pattern="^hear_"))
    app.add_handler(CallbackQueryHandler(handle_voice_sent, pattern="^voice_sent$"))
    app.add_handler(CallbackQueryHandler(on_callback))  # Keep existing callback handler for other buttons
    
    app.add_error_handler(error_handler)

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()