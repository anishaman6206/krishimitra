import os
from typing import Optional, Tuple

from app.config import settings

# --- Script ranges we care about (Unicode blocks) ---
# NOTE: We map scripts to a single "primary" language for MVP.
SCRIPT_LANG_MAP = {
    "devanagari": ("hi", "\u0900", "\u097F"),   # Hindi (also Marathi/Nepali but we treat as 'hi')
    "gurmukhi":   ("pa", "\u0A00", "\u0A7F"),   # Punjabi
    "gujarati":   ("gu", "\u0A80", "\u0AFF"),
    "bengali":    ("bn", "\u0980", "\u09FF"),   # Bengali/Assamese
    "oriya":      ("or", "\u0B00", "\u0B7F"),
    "tamil":      ("ta", "\u0B80", "\u0BFF"),
    "telugu":     ("te", "\u0C00", "\u0C7F"),
    "kannada":    ("kn", "\u0C80", "\u0CFF"),
    "malayalam":  ("ml", "\u0D00", "\u0D7F"),
    "arabic":     ("ur", "\u0600", "\u06FF"),   # Urdu
}

HINGLISH_HINT_WORDS = {
    "kya", "kaise", "nahi", "nahiin", "nhi", "haan", "bilkul",
    "paani", "beej", "khad", "daam", "mandi", "fasal", "bima",
    "sarkari", "yojana", "krishi", "kharif", "rabi", "bigha",
}

def _dominant_script(text: str) -> Optional[str]:
    counts = {}
    for name, (_code, start, end) in SCRIPT_LANG_MAP.items():
        lo, hi = ord(start), ord(end)
        counts[name] = sum(1 for ch in text if lo <= ord(ch) <= hi)
    # Latin letters count separately to decide en vs Hinglish
    latin_count = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    # pick max non-latin script if any
    best_script = max(counts, key=lambda k: counts[k]) if counts else None
    if best_script and counts[best_script] >= max(3, int(0.2 * len(text))):
        return best_script
    # otherwise it's mostly Latin (English or Hinglish)
    if latin_count >= max(3, int(0.2 * len(text))):
        return "latin"
    return None

def detect_lang(text: str) -> str:
    """
    Returns ISO-ish code ('en','hi','kn',...) or 'hi-Latn' for Hinglish.
    Very lightweight heuristic; good enough for MVP.
    """
    if not text or text.strip() == "":
        return "en"
    script = _dominant_script(text)
    if script and script != "latin":
        return SCRIPT_LANG_MAP[script][0]
    # Latin: check for Hinglish hint words
    tokens = {t.strip(".,!?;:()[]{}'\"").lower() for t in text.split()}
    if tokens & HINGLISH_HINT_WORDS:
        return "hi-Latn"
    return "en"

# ------------- OpenAI translation helpers (sync) -----------------

def _lang_name(code: str) -> Tuple[str, Optional[str]]:
    """Return (language name, script hint) for prompts."""
    mapping = {
        "en": ("English", None),
        "hi": ("Hindi", "Devanagari"),
        "hi-Latn": ("Hindi (romanized)", "Latin"),
        "pa": ("Punjabi", "Gurmukhi"),
        "gu": ("Gujarati", "Gujarati"),
        "bn": ("Bengali", "Bengali"),
        "or": ("Odia", "Odia"),
        "ta": ("Tamil", "Tamil"),
        "te": ("Telugu", "Telugu"),
        "kn": ("Kannada", "Kannada"),
        "ml": ("Malayalam", "Malayalam"),
        "ur": ("Urdu", "Arabic"),
    }
    return mapping.get(code, ("English", None))

def _openai_translate(text: str, src_code: str, tgt_code: str) -> str:
    """
    Use OpenAI chat model (from settings) to translate; return text only.
    Fallback: original text if the API call fails.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)

        src_name, _ = _lang_name(src_code)
        tgt_name, tgt_script = _lang_name(tgt_code)
        script_note = f" Use the {tgt_script} script." if tgt_script else ""

        system = (
            "You are a precise translator for Indian agricultural messages. "
            "Return ONLY the translated text, no quotes or explanations. "
            "Preserve numbers, units, dates, and crop names. "
            "Do not add extra sentences."
        )
        user = (
            f"Translate from {src_name} to {tgt_name}.{script_note}\n"
            "If the input mixes languages, translate the non-English parts and keep English words."
            "\n\nTEXT:\n" + text
        )

        resp = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            temperature=0.0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        out = resp.choices[0].message.content.strip()
        return out
    except Exception:
        # Fail safe: just return original
        return text

# ------------- Public API (used by pipeline) ---------------------

def translate_to_en(text: str) -> str:
    code = detect_lang(text)
    if code in ("en", None):
        return text
    return _openai_translate(text, src_code=code, tgt_code="en")

def translate_from_en(text: str, target: str) -> str:
    target = target or "en"
    if target == "en":
        return text
    return _openai_translate(text, src_code="en", tgt_code=target)

# ------------- CLI for quick tests -------------------------------
if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser(description="Language detect + OpenAI translate")
    p.add_argument("--text", required=True, help="Input text")
    p.add_argument("--to", default="en", help="Target code (en, hi, kn, te, ta, ml, bn, gu, or, pa, ur, hi-Latn)")
    args = p.parse_args()

    detected = detect_lang(args.text)
    if args.to == "auto-en":
        translated = translate_to_en(args.text)
        print(f"detect={detected}\n→ {translated}")
    else:
        translated = translate_from_en(args.text if detected == 'en' else translate_to_en(args.text), args.to)
        print(f"detect={detected}\n→ {translated}")
