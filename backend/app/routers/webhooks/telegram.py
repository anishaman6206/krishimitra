from fastapi import APIRouter, Request, HTTPException, Depends
from app.di import get_conversation_service, get_telegram_adapter, get_http
from core.services.conversation import ConversationService
from core.adapters.telegram import TelegramAdapter
import os
import json
import hashlib
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import tempfile

from backend.app.utils.voice_utils import opus_to_wav, sarvam_stt, sarvam_tts, wav_to_opus
VOICE_AVAILABLE = True

router = APIRouter(prefix="/webhooks/telegram", tags=["webhooks"])

# Environment
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    print("⚠️  Warning: TELEGRAM_BOT_TOKEN not set - webhook will only process but not respond")

API_BASE = f"https://api.telegram.org/bot{BOT_TOKEN}" if BOT_TOKEN else None

# Per-chat preferences (from old bot - useful feature)
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

# Voice message cache for TTS responses
import time
_voice_cache: Dict[str, Dict[str, Any]] = {}
_VOICE_TTL_SEC = 15 * 60  # 15 minutes


def _get_prefs(chat_id: int) -> ChatPrefs:
    """Get or create chat preferences."""
    if chat_id not in PREFS:
        PREFS[chat_id] = ChatPrefs()
    return PREFS[chat_id]

def _reply_keyboard():
    """Create reply keyboard with useful buttons."""
    return {
        "keyboard": [
            [{"text": "📍 Share location", "request_location": True}],
            [{"text": "Set crop"}, {"text": "Set horizon"}],
            [{"text": "Help"}, {"text": "My settings"}]
        ],
        "resize_keyboard": True,
        "one_time_keyboard": False
    }

def _followup_buttons(tool_notes: Dict[str, Any]) -> Optional[Dict]:
    """Create inline keyboard from tool results."""
    buttons = []
    
    # NDVI image button - add debugging
    ndvi_ql = tool_notes.get("ndvi_quicklook", {})
    print(f"🛰️ NDVI quicklook data: {ndvi_ql}")
    
    # Check if NDVI data is available
    if ndvi_ql.get("available") and (ndvi_ql.get("url") or ndvi_ql.get("bytes")):
        print(f"✅ Adding NDVI button - available: {ndvi_ql.get('available')}, url: {ndvi_ql.get('url')}, bytes: {ndvi_ql.get('bytes')}")
        buttons.append([{"text": "🛰️ View NDVI Image", "callback_data": "ndvi_img"}])
    else:
        print(f"❌ NDVI button not added - available: {ndvi_ql.get('available')}, url: {ndvi_ql.get('url')}, bytes: {ndvi_ql.get('bytes')}")
    
    # Horizon buttons (commented out for now)
    # buttons.append([
    #     {"text": "📅 Week 1", "callback_data": "hz_7"},
    #     {"text": "📅 Week 2", "callback_data": "hz_14"}
    # ])
    
    # Market selection (commented out for now)
    # buttons.append([{"text": "🏪 Change Market", "callback_data": "set_market"}])
    
    return {"inline_keyboard": buttons} if buttons else None

@router.post("")
async def telegram_webhook(
    req: Request,
    conversation_service: ConversationService = Depends(get_conversation_service),
    adapter: TelegramAdapter = Depends(get_telegram_adapter),
    http = Depends(get_http)
):
    """
    Unified webhook handler with clean architecture + rich features.
    """
    try:
        # 1) Parse the update
        update = await req.json()
        print(f"📱 Telegram webhook received: {update}")
        
        # Handle different update types
        if update.get("message"):
            return await _handle_message(update, conversation_service, adapter, http)
        elif update.get("callback_query"):
            return await _handle_callback(update, http)
        else:
            # Ignore other update types
            return {"ok": True}
            
    except json.JSONDecodeError:
        print("❌ Telegram webhook: Invalid JSON received")
        raise HTTPException(400, "Invalid JSON")
    except Exception as e:
        print(f"❌ Telegram webhook error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}, 500

async def _handle_message(update: Dict, conversation_service: ConversationService, adapter: TelegramAdapter, http):
    """Handle regular messages with full feature support."""
    message = update["message"]
    chat_id = message["chat"]["id"]
    prefs = _get_prefs(chat_id)
    
    # Handle different message types
    if message.get("location"):
        return await _handle_location(message, chat_id, http)
    elif message.get("voice"):
        return await _handle_voice(message, chat_id, conversation_service, adapter, http, prefs)
    elif message.get("text"):
        return await _handle_text(message, chat_id, conversation_service, adapter, http, prefs)
    
    return {"ok": True}

async def _handle_location(message: Dict, chat_id: int, http):
    """Handle location sharing."""
    prefs = _get_prefs(chat_id)
    location = message["location"]
    prefs.lat = float(location["latitude"])
    prefs.lon = float(location["longitude"])
    
    if API_BASE:
        await http.post(f"{API_BASE}/sendMessage", json={
            "chat_id": chat_id,
            "text": f"📍 Got your location: {prefs.lat:.5f}, {prefs.lon:.5f}\nNow I can provide local weather and satellite data!",
            "reply_markup": _reply_keyboard()
        })
    
    return {"ok": True}

async def _handle_voice(message: Dict, chat_id: int, conversation_service: ConversationService, 
                       adapter: TelegramAdapter, http, prefs: ChatPrefs):
    """Handle voice messages with full STT/TTS functionality."""
    if not API_BASE:
        return {"ok": True}
    
    try:
        # Show typing indicator
        await http.post(f"{API_BASE}/sendChatAction", json={
            "chat_id": chat_id,
            "action": "typing"
        })
        
        # Get voice file info
        voice = message["voice"]
        file_id = voice["file_id"]
        short_id = hashlib.md5(file_id.encode()).hexdigest()[:12]
        
        if VOICE_AVAILABLE:
            # Download and process voice with actual STT
            file_response = await http.get(f"{API_BASE}/getFile?file_id={file_id}")
            file_data = file_response.json()
            
            if file_data.get("ok"):
                file_path = file_data["result"]["file_path"]
                file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
                
                # Download voice file
                voice_response = await http.get(file_url)

                with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as ogg_f:
                    input_ogg = ogg_f.name
                    ogg_f.write(voice_response.content)

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_f:
                    input_wav = wav_f.name
                    await opus_to_wav(input_ogg, input_wav)

                try:
                    transcript, detected_lang = await sarvam_stt(input_wav)
                finally:
                    for p in (input_ogg, input_wav):
                        try:
                            os.remove(p)
                        except:
                            pass

                
                if not transcript.strip():
                    await http.post(f"{API_BASE}/sendMessage", json={
                        "chat_id": chat_id,
                        "text": "Sorry, I couldn't understand the audio. Please try again or type your message."
                    })
                    return {"ok": True}
                
                print(f"🎤 STT Result: '{transcript}' (Language: {detected_lang})")
                
            else:
                transcript = "[Voice message - could not download file]"
                detected_lang = "en"
        else:
            # Fallback when voice utilities not available
            transcript = "[Voice message - transcription not available]"
            detected_lang = "en"
        
        # Process through conversation service
        core_in = adapter.to_core({
            "message": {
                **message,
                "text": transcript  # Replace voice with transcript
            }
        })
        
        # Add preferences to metadata
        core_in.metadata.update({
            "crop": prefs.crop,
            "state": prefs.state,
            "district": prefs.district,
            "market": prefs.market,
            "horizon_days": prefs.horizon_days
        })
        
        # Set language hint from detected language
        core_in.lang_hint = detected_lang
        
        if prefs.lat and prefs.lon:
            core_in.geo = {"lat": prefs.lat, "lon": prefs.lon}
        
        # Get AI response
        core_out = await conversation_service.handle(core_in)
        
        # Update preferences with tool notes
        if core_out.tool_notes:
            prefs.last_tool_notes = core_out.tool_notes
        
        # Send response with voice option
        response_text = core_out.text or "I couldn't process your voice message."
        
        # Create buttons for voice response and follow-ups
        buttons = []
        
        # Voice response button (if TTS available)
        if VOICE_AVAILABLE:
            buttons.append([{"text": "🔊 Hear this answer", "callback_data": f"voice_{short_id}"}])
        
        # Add follow-up buttons from tool notes
        followup_keyboard = _followup_buttons(core_out.tool_notes or {})
        if followup_keyboard and followup_keyboard.get("inline_keyboard"):
            buttons.extend(followup_keyboard["inline_keyboard"])
        
        reply_markup = {"inline_keyboard": buttons} if buttons else None
        
        # Send text response
        await http.post(f"{API_BASE}/sendMessage", json={
            "chat_id": chat_id,
            "text": f"🎤 You said: \"{transcript}\"\n\n {response_text}",
            "reply_markup": reply_markup,
            "disable_web_page_preview": True
        })
        
        # Store answer for TTS generation if requested
        if VOICE_AVAILABLE:
            # Store in a simple cache for TTS callback
            _voice_cache[f"voice_{short_id}"] = {
                "text": response_text,
                "language": detected_lang,
                "chat_id": chat_id,
                "ts": time.time()
            }

        
        # Send attachments if any
        for attachment in core_out.attachments:
            if attachment.kind == "image":
                await http.post(f"{API_BASE}/sendPhoto", json={
                    "chat_id": chat_id,
                    "photo": attachment.url,
                    "caption": attachment.caption or ""
                })
        
    except Exception as e:
        print(f"Voice processing error: {e}")
        import traceback
        traceback.print_exc()
        
        if API_BASE:
            await http.post(f"{API_BASE}/sendMessage", json={
                "chat_id": chat_id,
                "text": "Sorry, I encountered an error processing your voice message. Please try typing instead."
            })
    
    return {"ok": True}

async def _handle_text(message: Dict, chat_id: int, conversation_service: ConversationService,
                      adapter: TelegramAdapter, http, prefs: ChatPrefs):
    """Handle text messages with command support."""
    text = message.get("text", "").strip()
    
    # Handle quick commands
    if text.startswith("/start"):
        return await _send_start_message(chat_id, http)
    elif text.startswith("/help"):
        return await _send_help_message(chat_id, http)
    elif text.startswith("/setcrop"):
        return await _handle_setcrop(text, chat_id, prefs, http)
    elif text.startswith("/sethorizon"):
        return await _handle_sethorizon(text, chat_id, prefs, http)
    elif text.startswith("/testndvi"):
        return await _handle_test_ndvi(chat_id, prefs, http)
    elif text.lower().startswith("set crop"):
        return await _handle_quick_setcrop(text, chat_id, prefs, http)
    elif text.lower().startswith("set market"):
        return await _handle_quick_setmarket(text, chat_id, prefs, http)
    elif text.lower() in ["help", "my settings"]:
        return await _send_settings(chat_id, prefs, http)
    
    # Regular AI query - show typing indicator
    if API_BASE:
        await http.post(f"{API_BASE}/sendChatAction", json={
            "chat_id": chat_id,
            "action": "typing"
        })
    
    try:
        # Convert to core message
        core_in = adapter.to_core({"message": message})
        
        # Add preferences to metadata
        core_in.metadata.update({
            "crop": prefs.crop,
            "state": prefs.state,
            "district": prefs.district,
            "market": prefs.market,
            "horizon_days": prefs.horizon_days
        })
        
        if prefs.lat and prefs.lon:
            core_in.geo = {"lat": prefs.lat, "lon": prefs.lon}
        
        # Process through conversation service
        core_out = await conversation_service.handle(core_in)
        
        # Update preferences with tool notes
        if core_out.tool_notes:
            prefs.last_tool_notes = core_out.tool_notes
        
        # Send response
        if API_BASE and core_out.text:
            # Create follow-up buttons from tool notes
            reply_markup = _followup_buttons(core_out.tool_notes or {})
            
            payload = {
                "chat_id": chat_id,
                "text": core_out.text,
                "disable_web_page_preview": True
            }
            
            if reply_markup:
                payload["reply_markup"] = reply_markup
            
            await http.post(f"{API_BASE}/sendMessage", json=payload)
            
            # Send any attachments
            for attachment in core_out.attachments:
                if attachment.kind == "image":
                    await http.post(f"{API_BASE}/sendPhoto", json={
                        "chat_id": chat_id,
                        "photo": attachment.url,
                        "caption": attachment.caption or ""
                    })
        
    except Exception as e:
        print(f"Text processing error: {e}")
        if API_BASE:
            await http.post(f"{API_BASE}/sendMessage", json={
                "chat_id": chat_id,
                "text": f"Sorry, I encountered an error: {str(e)}"
            })
    
    return {"ok": True}

async def _handle_callback(update: Dict, http):
    """Handle callback queries (button presses)."""
    print(f"🔔 callback received: {update.get('callback_query', {}).get('data')}")
    
    query = update["callback_query"]
    chat_id = query["message"]["chat"]["id"]
    data = query["data"]
    prefs = _get_prefs(chat_id)
    
    # Answer the callback query
    if API_BASE:
        await http.post(f"{API_BASE}/answerCallbackQuery", json={
            "callback_query_id": query["id"]
        })
    
    # Handle different callbacks
    if data == "ndvi_img":
        # Send NDVI image
        await _handle_ndvi_image_request(chat_id, query["message"]["message_id"], prefs, http)
            
    elif data.startswith("voice_"):
        # Handle TTS voice response
        print(f"🔊 Callback received for voice: {data}")
        print(f"🔊 VOICE_AVAILABLE: {VOICE_AVAILABLE}")
        print(f"🔊 Cache keys: {list(_voice_cache.keys())}")
        print(f"🔊 Data in cache: {data in _voice_cache}")
        if VOICE_AVAILABLE and data in _voice_cache:
            await _handle_tts_response(data, chat_id, http)
        else:
            print(f"❌ Voice not available or not in cache")
            await http.post(f"{API_BASE}/sendMessage", json={
                "chat_id": chat_id,
                "text": "🔊 Voice response not available"
            })
            
    elif data == "hz_7":
        prefs.horizon_days = 7
        if API_BASE:
            await http.post(f"{API_BASE}/sendMessage", json={
                "chat_id": chat_id,
                "text": "✅ Horizon set to 7 days"
            })
            
    elif data == "hz_14":
        prefs.horizon_days = 14
        if API_BASE:
            await http.post(f"{API_BASE}/sendMessage", json={
                "chat_id": chat_id,
                "text": "✅ Horizon set to 14 days"
            })
            
    elif data == "set_market":
        if API_BASE:
            await http.post(f"{API_BASE}/sendMessage", json={
                "chat_id": chat_id,
                "text": "Send me your market like:\nSet market: Karnataka, Bangalore, Ramanagara"
            })
    
    return {"ok": True}

async def _handle_ndvi_image_request(chat_id: int, message_id: int, prefs: ChatPrefs, http):
    """Handle NDVI image request from callback."""
    if not API_BASE:
        return
    
    try:
        # Check if we have location data
        if not prefs.lat or not prefs.lon:
            await http.post(f"{API_BASE}/sendMessage", json={
                "chat_id": chat_id,
                "text": "📍 Please share your location first to view NDVI satellite imagery.",
                "reply_to_message_id": message_id
            })
            return
        
        # Show loading message
        loading_response = await http.post(f"{API_BASE}/sendMessage", json={
            "chat_id": chat_id,
            "text": "🛰️ Generating NDVI satellite image...",
            "reply_to_message_id": message_id
        })
        
        try:
            loading_msg_id = loading_response.json().get("result", {}).get("message_id")
        except:
            loading_msg_id = None
        
        # Generate NDVI quicklook
        from backend.app.tools.sentinel_cached import ndvi_quicklook_cached
        
        img_bytes = await ndvi_quicklook_cached(
            lat=prefs.lat,
            lon=prefs.lon,
            aoi_km=1.0,
            recent_days=20
        )
        
        # Delete loading message
        if loading_msg_id:
            try:
                await http.post(f"{API_BASE}/deleteMessage", json={
                    "chat_id": chat_id,
                    "message_id": loading_msg_id
                })
            except:
                pass  # Ignore if deletion fails
        
        if img_bytes:
            # Send the image as file upload
            import io
            import tempfile
            import os
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_file.write(img_bytes)
                tmp_file_path = tmp_file.name
            
            try:
                # Send photo using file upload with httpx
                with open(tmp_file_path, "rb") as photo_file:
                    files = {"photo": ("ndvi_image.png", photo_file, "image/png")}
                    data = {
                        "chat_id": str(chat_id),
                        "caption": f"🛰️ NDVI Satellite Image\n📍 Location: {prefs.lat:.4f}, {prefs.lon:.4f}\n🌱 Green areas show healthy vegetation\n📅 Recent imagery (last 20 days)",
                        "reply_to_message_id": str(message_id)
                    }
                    
                    # Use the existing http client for file upload
                    response = await http.post(f"{API_BASE}/sendPhoto", data=data, files=files)
                    
                    if response.status_code != 200:
                        print(f"❌ Failed to send NDVI image: {response.status_code} - {response.text}")
                        await http.post(f"{API_BASE}/sendMessage", json={
                            "chat_id": chat_id,
                            "text": "❌ Error sending NDVI image. Please try again.",
                            "reply_to_message_id": message_id
                        })
                    else:
                        print(f"✅ NDVI image sent successfully")
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
        else:
            await http.post(f"{API_BASE}/sendMessage", json={
                "chat_id": chat_id,
                "text": "❌ Could not generate NDVI image. Satellite data may not be available for your location or there might be too much cloud cover.",
                "reply_to_message_id": message_id
            })
            
    except Exception as e:
        print(f"❌ Error handling NDVI image request: {e}")
        import traceback
        traceback.print_exc()
        
        await http.post(f"{API_BASE}/sendMessage", json={
            "chat_id": chat_id,
            "text": "❌ Error generating NDVI image. Please try again later.",
            "reply_to_message_id": message_id
        })

async def _handle_test_ndvi(chat_id: int, prefs: ChatPrefs, http):
    """Test NDVI functionality directly."""
    if not API_BASE:
        return {"ok": True}
    
    if not prefs.lat or not prefs.lon:
        await http.post(f"{API_BASE}/sendMessage", json={
            "chat_id": chat_id,
            "text": "📍 Share location first, then try /testndvi"
        })
        return {"ok": True}
    
    await _handle_ndvi_image_request(chat_id, 0, prefs, http)  # message_id = 0 for test
    return {"ok": True}

def _to_sarvam_lang(code: str) -> str:
    if "-" in code:
        return code
    return {
        "en": "en-IN",
        "hi": "hi-IN",
        "bn": "bn-IN",
        "ta": "ta-IN",
        "te": "te-IN",
        "gu": "gu-IN",
        "kn": "kn-IN",
        "ml": "ml-IN",
        "mr": "mr-IN",
        "pa": "pa-IN",
        "or": "or-IN",
    }.get(code, "en-IN")


async def _handle_tts_response(voice_key: str, chat_id: int, http):
    """Generate and send TTS voice response (OGG/Opus for Telegram)."""
    print(f"🔊 Starting TTS for voice_key: {voice_key}, chat_id: {chat_id}")
    try:
        voice_data = _voice_cache.get(voice_key)
        print(f"🔊 Voice cache lookup result: {voice_data}")
        if not voice_data:
            print("❌ Voice data not found in cache")
            await http.post(f"{API_BASE}/sendMessage", json={
                "chat_id": chat_id,
                "text": "Voice response expired. Please ask again."
            })
            return

        text = voice_data["text"]
        language = voice_data["language"]
        print(f"🔊 Text: {text[:50]}..., Language: {language}")

        # Show upload indicator
        await http.post(f"{API_BASE}/sendChatAction", json={"chat_id": chat_id, "action": "upload_voice"})

        short_id = voice_key.replace("voice_", "")
        tmp_wav = f"tts_{chat_id}_{short_id}.wav"
        tmp_ogg = f"tts_{chat_id}_{short_id}.ogg"

        # Normalize language to Sarvam
        lang_code = language
        if "-" not in lang_code:
            lang_code = "hi-IN" if lang_code == "hi" else ("en-IN" if lang_code == "en" else "en-IN")

        print(f"🔊 About to call sarvam_tts with lang_code: {lang_code}")
        # 1) Synthesize WAV using Sarvam
        await sarvam_tts(text, tmp_wav, lang_code)
        print(f"🔊 sarvam_tts completed, WAV file created: {tmp_wav}")

        print(f"🔊 About to convert WAV to OGG")
        # 2) Convert WAV -> OGG/Opus (Telegram expects Opus-in-OGG for /sendVoice)
        await wav_to_opus(tmp_wav, tmp_ogg)
        print(f"🔊 wav_to_opus completed, OGG file created: {tmp_ogg}")

        print(f"🔊 About to send voice message")
        # 3) Upload OGG as voice
        with open(tmp_ogg, "rb") as f:
            files = {"voice": ("voice.ogg", f, "audio/ogg")}
            data = {"chat_id": chat_id, "caption": "🔊 Voice response"}
            resp = await http.post(f"{API_BASE}/sendVoice", files=files, data=data)
            try:
                resp.raise_for_status()
                print(f"🔊 Voice message sent successfully")
            except Exception:
                print(f"❌ /sendVoice failed: {resp.status_code} {resp.text}")
                raise

        # Cleanup & remove cache entry
        try:
            os.remove(tmp_wav)
            os.remove(tmp_ogg)
        except Exception:
            pass
        _voice_cache.pop(voice_key, None)

        print("✅ TTS voice sent")

    except Exception as e:
        print(f"TTS error: {e}\n{traceback.format_exc()}")
        fallback_text = _voice_cache.get(voice_key, {}).get("text", "Voice response unavailable")
        await http.post(f"{API_BASE}/sendMessage", json={
            "chat_id": chat_id,
            "text": f"Sorry, I couldn't generate the voice response. Here's the text instead:\n\n{fallback_text}"
        })


# Command handlers
async def _send_start_message(chat_id: int, http):
    """Send start message."""
    if not API_BASE:
        return {"ok": True}
    
    await http.post(f"{API_BASE}/sendMessage", json={
        "chat_id": chat_id,
        "text": (
            "👋 *Namaste / Hello! I’m KrishiMitra AI.*\n\n"
        "Type *or speak* in any language — I’ll reply in the same.\n"
        "आप *किसी भी भाषा* में लिखें या बोलें — मैं उसी भाषा में जवाब दूँगा।\n\n"
        "_Examples:_\n"
        "• *English:* Where can I get affordable credit…?\n"
        "• *Hinglish:* Kal barish hogi? irrigation kab karu?\n"
        "• *Hindi:* क्या मुझे इस हफ्ते टमाटर बेचना चाहिए?\n\n"
        "🌱 You can also send a clear photo of a plant leaf for instant disease detection and cure tips.\n\n"
        "Tip: tap *📍 Share location* for local prices, weather & NDVI."
        ),
        "reply_markup": _reply_keyboard(),
        "parse_mode": "Markdown"
    })
    return {"ok": True}

async def _send_help_message(chat_id: int, http):
    """Send help message."""
    if not API_BASE:
        return {"ok": True}
    
    await http.post(f"{API_BASE}/sendMessage", json={
        "chat_id": chat_id,
        "text": (
            "🤖 **KrishiMitra Commands:**\n\n"
            "/setcrop Tomato - Set your crop\n"
            "/sethorizon 7 - Set forecast days (7 or 14)\n\n"
            "**Quick actions:**\n"
            "• Set crop: Tomato\n"
            "• Set market: Karnataka, Bangalore, Ramanagara\n"
            "• Share location for local data\n\n"
            "**Or just ask:** \"Should I sell tomatoes now?\""
        ),
        "parse_mode": "Markdown"
    })
    return {"ok": True}

async def _handle_setcrop(text: str, chat_id: int, prefs: ChatPrefs, http):
    """Handle /setcrop command."""
    parts = text.split(" ", 1)
    if len(parts) < 2:
        if API_BASE:
            await http.post(f"{API_BASE}/sendMessage", json={
                "chat_id": chat_id,
                "text": "Usage: /setcrop Tomato"
            })
    else:
        prefs.crop = parts[1].strip()
        if API_BASE:
            await http.post(f"{API_BASE}/sendMessage", json={
                "chat_id": chat_id,
                "text": f"✅ Crop set to: {prefs.crop}"
            })
    return {"ok": True}

async def _handle_sethorizon(text: str, chat_id: int, prefs: ChatPrefs, http):
    """Handle /sethorizon command."""
    parts = text.split(" ", 1)
    if len(parts) < 2 or not parts[1].isdigit() or int(parts[1]) not in (7, 14):
        if API_BASE:
            await http.post(f"{API_BASE}/sendMessage", json={
                "chat_id": chat_id,
                "text": "Usage: /sethorizon 7 (or 14)"
            })
    else:
        prefs.horizon_days = int(parts[1])
        if API_BASE:
            await http.post(f"{API_BASE}/sendMessage", json={
                "chat_id": chat_id,
                "text": f"✅ Horizon set to: {prefs.horizon_days} days"
            })
    return {"ok": True}

async def _handle_quick_setcrop(text: str, chat_id: int, prefs: ChatPrefs, http):
    """Handle 'Set crop: Tomato' format."""
    try:
        crop = text.split(":", 1)[1].strip()
        prefs.crop = crop
        if API_BASE:
            await http.post(f"{API_BASE}/sendMessage", json={
                "chat_id": chat_id,
                "text": f"✅ Crop set to: {prefs.crop}"
            })
    except:
        if API_BASE:
            await http.post(f"{API_BASE}/sendMessage", json={
                "chat_id": chat_id,
                "text": "Format: Set crop: Tomato"
            })
    return {"ok": True}

async def _handle_quick_setmarket(text: str, chat_id: int, prefs: ChatPrefs, http):
    """Handle 'Set market: State, District, Market' format."""
    try:
        market_info = text.split(":", 1)[1].strip()
        parts = [p.strip() for p in market_info.split(",")]
        if len(parts) >= 1:
            prefs.state = parts[0]
        if len(parts) >= 2:
            prefs.district = parts[1]
        if len(parts) >= 3:
            prefs.market = parts[2]
        
        if API_BASE:
            await http.post(f"{API_BASE}/sendMessage", json={
                "chat_id": chat_id,
                "text": (
                    f"✅ Market updated:\n"
                    f"State: {prefs.state}\n"
                    f"District: {prefs.district}\n"
                    f"Market: {prefs.market}"
                )
            })
    except:
        if API_BASE:
            await http.post(f"{API_BASE}/sendMessage", json={
                "chat_id": chat_id,
                "text": "Format: Set market: Karnataka, Bangalore, Ramanagara"
            })
    return {"ok": True}

async def _send_settings(chat_id: int, prefs: ChatPrefs, http):
    """Send current settings."""
    if not API_BASE:
        return {"ok": True}
    
    location = f"{prefs.lat:.4f}, {prefs.lon:.4f}" if prefs.lat and prefs.lon else "Not set"
    
    await http.post(f"{API_BASE}/sendMessage", json={
        "chat_id": chat_id,
        "text": (
            f"⚙️ **Your Settings:**\n\n"
            f"🌾 Crop: {prefs.crop or 'Not set'}\n"
            f"📍 Location: {location}\n"
            f"🏪 State: {prefs.state or 'Not set'}\n"
            f"🏪 District: {prefs.district or 'Not set'}\n"
            f"🏪 Market: {prefs.market or 'Not set'}\n"
            f"📅 Horizon: {prefs.horizon_days} days"
        ),
        "parse_mode": "Markdown"
    })
    return {"ok": True}
