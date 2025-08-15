import os
import json
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from pathlib import Path
import httpx

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

async def _ask_backend(prompt: str, prefs: ChatPrefs, debug: bool = False) -> Dict[str, Any]:
    payload = {
        "text": prompt,
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
    except Exception:
        pass


    try:
        resp = await _ask_backend(text, prefs)
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

def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("setcrop", setcrop_cmd))
    app.add_handler(CommandHandler("sethorizon", sethorizon_cmd))

    app.add_handler(MessageHandler(filters.LOCATION, on_location))
    app.add_handler(MessageHandler(filters.TEXT & filters.Regex(r"^Set market:"), on_text_market))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_error_handler(error_handler)

    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
