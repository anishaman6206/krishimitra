# core/services/conversation.py
import re
from typing import Optional, Dict, Any

from .rag import RAGService
from .pricing import PricingService
from .weather import WeatherService
from .ndvi import NDVIService
from .mandi import MandiService
from ..models.io import ChannelMessageIn, ChannelMessageOut

from backend.app.schemas import AskRequest, Geo
from backend.app.services.pipeline import answer as pipeline_answer
from backend.app.tools.lang import detect_lang  # translate_* handled in pipeline

# Common crop names for text extraction
CROP_KEYWORDS = {
    'tomato': 'Tomato',
    'tomatoes': 'Tomato',
    'onion': 'Onion',
    'onions': 'Onion', 
    'potato': 'Potato',
    'potatoes': 'Potato',
    'rice': 'Rice',
    'wheat': 'Wheat',
    'maize': 'Maize',
    'corn': 'Maize',
    'cotton': 'Cotton',
    'sugarcane': 'Sugarcane',
    'soybean': 'Soybean',
    'soya': 'Soybean',
    'chilli': 'Chilli',
    'chili': 'Chilli',
    'pepper': 'Chilli',
    'turmeric': 'Turmeric',
    'ginger': 'Ginger',
    'garlic': 'Garlic',
    'cabbage': 'Cabbage',
    'cauliflower': 'Cauliflower',
    'carrot': 'Carrot',
    'carrots': 'Carrot',
    'brinjal': 'Brinjal',
    'eggplant': 'Brinjal',
    'okra': 'Okra',
    'ladyfinger': 'Okra',
    'cucumber': 'Cucumber',
    'bottle gourd': 'Bottle Gourd',
    'ridge gourd': 'Ridge Gourd',
    'bitter gourd': 'Bitter Gourd',
    'pumpkin': 'Pumpkin',
    'watermelon': 'Watermelon',
    'mango': 'Mango',
    'mangoes': 'Mango',
    'banana': 'Banana',
    'bananas': 'Banana',
    'grapes': 'Grapes',
    'apple': 'Apple',
    'apples': 'Apple',
    'orange': 'Orange',
    'oranges': 'Orange',
    'lemon': 'Lemon',
    'lemons': 'Lemon',
    'papaya': 'Papaya',
    'guava': 'Guava',
}

# Common Indian cities with coordinates for location extraction
CITY_COORDINATES = {
    'delhi': {'lat': 28.6139, 'lon': 77.2090, 'state': 'Delhi'},
    'mumbai': {'lat': 19.0760, 'lon': 72.8777, 'state': 'Maharashtra'},
    'bangalore': {'lat': 12.9716, 'lon': 77.5946, 'state': 'Karnataka'},
    'bengaluru': {'lat': 12.9716, 'lon': 77.5946, 'state': 'Karnataka'},
    'hyderabad': {'lat': 17.3850, 'lon': 78.4867, 'state': 'Telangana'},
    'ahmedabad': {'lat': 23.0225, 'lon': 72.5714, 'state': 'Gujarat'},
    'chennai': {'lat': 13.0827, 'lon': 80.2707, 'state': 'Tamil Nadu'},
    'kolkata': {'lat': 22.5726, 'lon': 88.3639, 'state': 'West Bengal'},
    'pune': {'lat': 18.5204, 'lon': 73.8567, 'state': 'Maharashtra'},
    'jaipur': {'lat': 26.9124, 'lon': 75.7873, 'state': 'Rajasthan'},
    'lucknow': {'lat': 26.8467, 'lon': 80.9462, 'state': 'Uttar Pradesh'},
    'kanpur': {'lat': 26.4499, 'lon': 80.3319, 'state': 'Uttar Pradesh'},
    'nagpur': {'lat': 21.1458, 'lon': 79.0882, 'state': 'Maharashtra'},
    'indore': {'lat': 22.7196, 'lon': 75.8577, 'state': 'Madhya Pradesh'},
    'thane': {'lat': 19.2183, 'lon': 72.9781, 'state': 'Maharashtra'},
    'bhopal': {'lat': 23.2599, 'lon': 77.4126, 'state': 'Madhya Pradesh'},
    'visakhapatnam': {'lat': 17.6868, 'lon': 83.2185, 'state': 'Andhra Pradesh'},
    'pimpri': {'lat': 18.6298, 'lon': 73.7997, 'state': 'Maharashtra'},
    'patna': {'lat': 25.5941, 'lon': 85.1376, 'state': 'Bihar'},
    'vadodara': {'lat': 22.3072, 'lon': 73.1812, 'state': 'Gujarat'},
    'ludhiana': {'lat': 30.9010, 'lon': 75.8573, 'state': 'Punjab'},
    'agra': {'lat': 27.1767, 'lon': 78.0081, 'state': 'Uttar Pradesh'},
    'nashik': {'lat': 19.9975, 'lon': 73.7898, 'state': 'Maharashtra'},
    'faridabad': {'lat': 28.4089, 'lon': 77.3178, 'state': 'Haryana'},
    'meerut': {'lat': 28.9845, 'lon': 77.7064, 'state': 'Uttar Pradesh'},
    'rajkot': {'lat': 22.3039, 'lon': 70.8022, 'state': 'Gujarat'},
    'kalyan': {'lat': 19.2437, 'lon': 73.1355, 'state': 'Maharashtra'},
    'vasai': {'lat': 19.4912, 'lon': 72.8054, 'state': 'Maharashtra'},
    'varanasi': {'lat': 25.3176, 'lon': 82.9739, 'state': 'Uttar Pradesh'},
    'srinagar': {'lat': 34.0837, 'lon': 74.7973, 'state': 'Jammu and Kashmir'},
    'aurangabad': {'lat': 19.8762, 'lon': 75.3433, 'state': 'Maharashtra'},
    'dhanbad': {'lat': 23.7957, 'lon': 86.4304, 'state': 'Jharkhand'},
    'amritsar': {'lat': 31.6340, 'lon': 74.8723, 'state': 'Punjab'},
    'navi mumbai': {'lat': 19.0330, 'lon': 73.0297, 'state': 'Maharashtra'},
    'allahabad': {'lat': 25.4358, 'lon': 81.8463, 'state': 'Uttar Pradesh'},
    'prayagraj': {'lat': 25.4358, 'lon': 81.8463, 'state': 'Uttar Pradesh'},
    'ranchi': {'lat': 23.3441, 'lon': 85.3096, 'state': 'Jharkhand'},
    'howrah': {'lat': 22.5958, 'lon': 88.2636, 'state': 'West Bengal'},
    'coimbatore': {'lat': 11.0168, 'lon': 76.9558, 'state': 'Tamil Nadu'},
    'jabalpur': {'lat': 23.1815, 'lon': 79.9864, 'state': 'Madhya Pradesh'},
    'gwalior': {'lat': 26.2183, 'lon': 78.1828, 'state': 'Madhya Pradesh'},
    'vijayawada': {'lat': 16.5062, 'lon': 80.6480, 'state': 'Andhra Pradesh'},
    'jodhpur': {'lat': 26.2389, 'lon': 73.0243, 'state': 'Rajasthan'},
    'madurai': {'lat': 9.9252, 'lon': 78.1198, 'state': 'Tamil Nadu'},
    'raipur': {'lat': 21.2514, 'lon': 81.6296, 'state': 'Chhattisgarh'},
    'kota': {'lat': 25.2138, 'lon': 75.8648, 'state': 'Rajasthan'},
    'chandigarh': {'lat': 30.7333, 'lon': 76.7794, 'state': 'Chandigarh'},
    'guwahati': {'lat': 26.1445, 'lon': 91.7362, 'state': 'Assam'},
}

class ConversationService:
    """Central conversation orchestrator (delegates to pipeline)."""

    def __init__(self,
                 rag: RAGService,
                 pricing: PricingService,
                 weather: WeatherService,
                 ndvi: NDVIService,
                 mandi: MandiService):
        self.rag = rag
        self.pricing = pricing
        self.weather = weather
        self.ndvi = ndvi
        self.mandi = mandi

    def _extract_crop_from_text(self, text: str) -> Optional[str]:
        """Extract crop name from user text query."""
        if not text:
            return None
        
        text_lower = text.lower()
        print(f"🔍 Analyzing text for crops: '{text_lower}'")
        
        # Look for crop keywords in the text
        for keyword, crop_name in CROP_KEYWORDS.items():
            if keyword in text_lower:
                print(f"🌾 ✅ Found crop keyword: '{keyword}' -> '{crop_name}'")
                return crop_name
        
        print(f"🌾 ❌ No crop keywords found in: '{text_lower}'")
        return None

    def _extract_location_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract location coordinates from user text."""
        if not text:
            return None
        
        text_lower = text.lower()
        print(f"🌍 Analyzing text for location: '{text_lower}'")
        
        # Look for city names in the text
        for city_name, coords in CITY_COORDINATES.items():
            if city_name in text_lower:
                print(f"🌍 ✅ Found city: '{city_name}' -> {coords}")
                return coords
        
        print(f"🌍 ❌ No known cities found in: '{text_lower}'")
        return None

    def _request_location(self, lang: str) -> ChannelMessageOut:
        """Request user to provide location for agricultural data."""
        if lang == "hi":
            text = """🌍 कृपया अपना स्थान साझा करें।

मुझे आपके लिए सटीक जानकारी देने के लिए आपका स्थान चाहिए:
• 🌾 आपके क्षेत्र के फसल दाम
• 🌦️ मौसम पूर्वानुमान
• 🛰️ NDVI उपग्रह डेटा
• 💰 बेचने की सलाह

कृपया अपना स्थान साझा करें या शहर/जिला बताएं।"""
        else:
            text = """🌍 Please share your location.

I need your location to provide accurate information:
• 🌾 Local crop prices
• 🌦️ Weather forecasts
• 🛰️ NDVI satellite data
• 💰 Selling advice

Please share your location or tell me your city/district."""

        follow_ups = []
        if lang == "hi":
            follow_ups = [
                "मैं दिल्ली में हूं",
                "मैं मुंबई में हूं", 
                "मैं बैंगलोर में हूं",
                "मैं पुणे में हूं"
            ]
        else:
            follow_ups = [
                "I'm in Delhi",
                "I'm in Mumbai",
                "I'm in Bangalore", 
                "I'm in Pune"
            ]

        return ChannelMessageOut(
            text=text,
            followups=follow_ups,
            tool_notes={"location_required": True, "reason": "Location needed for agricultural data"}
        )

    async def handle(self, msg: ChannelMessageIn) -> ChannelMessageOut:
        try:
            ask_req = self._to_ask_request(msg)
            
            # LOCATION-FIRST REQUIREMENT: Check if location is provided
            if not ask_req.geo or ask_req.geo.lat is None or ask_req.geo.lon is None:
                return self._request_location(msg.lang_hint or "en")
            
            response = await pipeline_answer(ask_req)
            return self._to_channel_message_out(response, msg.channel)
        except Exception as e:
            print(f"Conversation service error: {e}")
            return ChannelMessageOut(text=f"Sorry, I encountered an error: {e}", tool_notes={"error": str(e)})

    def _to_ask_request(self, msg: ChannelMessageIn) -> AskRequest:
        text = msg.text
        if not text and msg.voice_url:
            text = "[Voice message - transcription not yet implemented]"

        # Extract location from text first, then check provided geo
        extracted_location = self._extract_location_from_text(text) if text else None
        
        geo = None
        if extracted_location:
            geo = Geo(lat=extracted_location['lat'], lon=extracted_location['lon'])
            print(f"🌍 Using extracted location: {extracted_location}")
        elif msg.geo:
            geo = Geo(lat=msg.geo.get("lat"), lon=msg.geo.get("lon"))
            print(f"🌍 Using provided geo: {msg.geo}")

        lang = msg.lang_hint
        if not lang and text:
            try:
                lang = detect_lang(text)
            except Exception:
                lang = "en"

        md = msg.metadata or {}
        
        # Try to extract crop from text first, fallback to metadata
        extracted_crop = self._extract_crop_from_text(text) if text else None
        crop = extracted_crop or md.get("crop")
        
        # Extract state from location if available
        state = md.get("state")
        if extracted_location and not state:
            state = extracted_location.get('state')
        
        print(f"🌾 Crop resolution: extracted='{extracted_crop}', metadata='{md.get('crop')}', final='{crop}'")
        print(f"🌍 Location resolution: extracted='{extracted_location}', final_geo='{geo}', state='{state}'")
        
        ask_request = AskRequest(
            text=text or "",
            lang=lang or "en",
            geo=geo,
            crop=crop,
            state=state,
            district=md.get("district"),
            market=md.get("market"),
            variety=md.get("variety"),
            grade=md.get("grade"),
            debug=False,
        )
        
        # 🐛 DEBUG: Print the complete request to see what's being sent to pipeline
        print(f"🔍 DEBUG AskRequest created:")
        print(f"   text: '{ask_request.text}'")
        print(f"   crop: '{ask_request.crop}'")
        print(f"   state: '{ask_request.state}'")
        print(f"   district: '{ask_request.district}'")
        print(f"   market: '{ask_request.market}'")
        print(f"   geo: {ask_request.geo}")
        print(f"   📊 Pipeline tool trigger conditions:")
        print(f"      Price tools will run if: crop='{bool(ask_request.crop)}' OR state='{bool(ask_request.state)}' OR district='{bool(ask_request.district)}' OR market='{bool(ask_request.market)}'")
        print(f"      Weather/NDVI will run if: geo='{bool(ask_request.geo and ask_request.geo.lat and ask_request.geo.lon)}'")
        
        return ask_request

    def _to_channel_message_out(self, pipeline_response, channel: str) -> ChannelMessageOut:
        if hasattr(pipeline_response, "answer"):
            answer = pipeline_response.answer
            sources = pipeline_response.sources or []
            tool_notes = pipeline_response.tool_notes or {}
            follow_ups = pipeline_response.follow_ups or []
        else:
            answer = pipeline_response.get("answer", "")
            sources = pipeline_response.get("sources", [])
            tool_notes = pipeline_response.get("tool_notes", {})
            follow_ups = pipeline_response.get("follow_ups", [])

        return ChannelMessageOut(text=answer, followups=follow_ups, tool_notes=tool_notes)
