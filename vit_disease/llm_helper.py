# vit_disease/llm_helper.py
from typing import Optional
import os
import openai
from langchain_openai import ChatOpenAI
from backend.app.utils.cache import get_json, set_json  # your in-memory cache
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

async def generate_disease_info(disease_name: str, lang: str = "en") -> str:
    """
    Returns a short, farmer-friendly summary (what it is, symptoms, and simple treatment).
    Cached by (disease, lang).
    """
    key = f"disease_info:{disease_name.strip().lower()}:{lang}"
    cached = await get_json(key, "default")
    if cached:
        return cached

    sys = (
        "You are an agronomy advisor. Explain plant diseases in simple language. "
        "Keep it practical, short, and safe. Avoid brand names; prefer generic actives. "
        "If info is uncertain, say so."
    )
    user = (
        f"Language: {lang}\n"
        f"Disease: {disease_name}\n\n"
        "Write 4 short bullets:\n"
        "• What it is\n"
        "• Key symptoms to check\n"
        "• Immediate low-cost action\n"
        "• Simple treatment (generic pesticide/fungicide name & dosage)\n"
    )

    out = _llm.invoke([{"role":"system","content":sys},{"role":"user","content":user}]).content.strip()
    await set_json(key, out, "default")
    return out
