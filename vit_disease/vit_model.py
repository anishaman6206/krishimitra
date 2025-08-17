from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch
import os

MODEL_DIR = "./vit-plant-disease"

# Load the fine-tuned feature extractor and model
feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_DIR)
model = ViTForImageClassification.from_pretrained(MODEL_DIR)

def predict_disease(image_path):
    """
    Predict plant disease from a leaf image using the fine-tuned ViT.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_idx = outputs.logits.argmax(-1).item()

    # Map index to class label
    label = model.config.id2label[predicted_class_idx]
    return label
