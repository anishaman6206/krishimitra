# vit_disease/vit_model.py
import os
from pathlib import Path
import torch
from PIL import Image

from transformers import AutoImageProcessor, AutoModelForImageClassification

# --- Resolve model directory robustly ---
HERE = Path(__file__).resolve().parent
# prefer env var if provided, otherwise use folder next to this file
MODEL_DIR = Path(os.getenv("VIT_MODEL_DIR", str(HERE / "vit-plant-disease"))).resolve()

def _has(path: Path, name: str) -> bool:
    return (path / name).exists()

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"[vit_model] MODEL_DIR does not exist: {MODEL_DIR}\n"
        "Set VIT_MODEL_DIR in your .env to the folder that contains "
        "config.json, preprocessor_config.json (or image_processor_config.json), "
        "and model.safetensors/pytorch_model.bin."
    )

# sanity checks
if not (_has(MODEL_DIR, "model.safetensors") or _has(MODEL_DIR, "pytorch_model.bin")):
    raise FileNotFoundError(f"[vit_model] No model weights found in {MODEL_DIR}")

# Prefer local image processor; if missing, fall back to a base preprocessor id
FALLBACK_PREPROC_ID = os.getenv("VIT_PREPROCESSOR_ID", "google/vit-base-patch16-224-in21k")
try:
    image_processor = AutoImageProcessor.from_pretrained(str(MODEL_DIR), local_files_only=True)
except Exception:
    image_processor = AutoImageProcessor.from_pretrained(FALLBACK_PREPROC_ID)

model = AutoModelForImageClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
model.eval()
id2label = model.config.id2label if hasattr(model.config, "id2label") else None

@torch.inference_mode()
def predict_disease(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=img, return_tensors="pt")
    logits = model(**inputs).logits
    pred_id = int(torch.argmax(logits, dim=-1))
    return id2label.get(pred_id, str(pred_id)) if id2label else str(pred_id)
