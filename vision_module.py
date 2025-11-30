"""
vision_module.py

Reusable image captioning + object detection module.
Works in:
- Jupyter / Colab (no argparse required)
- Streamlit
- CLI script (import this module)

Models:
- BLIP (captioning)
- DETR (object detection)
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    DetrImageProcessor,
    DetrForObjectDetection,
)

# -----------------------------
# Singleton Model Loader
# -----------------------------
class Models:
    caption_processor = None
    caption_model = None
    detection_processor = None
    detection_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Load Caption Model
# -----------------------------
def load_caption_model(model_name="Salesforce/blip-image-captioning-base"):
    if Models.caption_model is None:
        print(f"Loading caption model: {model_name} on {Models.device}")
        Models.caption_processor = BlipProcessor.from_pretrained(model_name)
        Models.caption_model = BlipForConditionalGeneration.from_pretrained(model_name).to(Models.device)
    return Models.caption_processor, Models.caption_model


# -----------------------------
# Load Detection Model
# -----------------------------
def load_detection_model(model_name="facebook/detr-resnet-50"):
    if Models.detection_model is None:
        print(f"Loading detection model: {model_name} on {Models.device}")
        Models.detection_processor = DetrImageProcessor.from_pretrained(model_name)
        Models.detection_model = DetrForObjectDetection.from_pretrained(model_name).to(Models.device)
    return Models.detection_processor, Models.detection_model


# -----------------------------
# Caption Image
# -----------------------------
def generate_caption(image_pil, max_length=30, num_beams=3):
    proc, model = load_caption_model()
    inputs = proc(images=image_pil, return_tensors="pt").to(Models.device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_length=max_length, num_beams=num_beams)

    caption = proc.decode(out_ids[0], skip_special_tokens=True)
    return caption


# -----------------------------
# Object Detection
# -----------------------------
def detect_objects(image_pil, threshold=0.5):
    proc, model = load_detection_model()
    inputs = proc(images=image_pil, return_tensors="pt").to(Models.device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image_pil.size[::-1]])
    results = proc.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections.append({
            "label": model.config.id2label[label.item()],
            "score": float(score),
            "box": [float(x) for x in box.tolist()],
        })
    return detections


# -----------------------------
# Draw Caption + Detection Boxes
# -----------------------------
def draw_annotations(image_pil, caption, detections):
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Draw caption bar
    W, H = img.size
    draw.rectangle([(0, 0), (W, 35)], fill="black")
    draw.text((10, 8), caption, fill="white", font=font)

    # Boxes
    for d in detections:
        x0, y0, x1, y1 = d["box"]
        label = f"{d['label']} {d['score']:.2f}"

        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        draw.text((x0, y0 - 12), label, fill="red", font=font)

    return img


# -----------------------------
# Main processing function
# -----------------------------
def analyze_image(image_path):
    image_pil = Image.open(image_path).convert("RGB")

    caption = generate_caption(image_pil)
    detections = detect_objects(image_pil)
    annotated = draw_annotations(image_pil, caption, detections)

    return caption, detections, annotated
