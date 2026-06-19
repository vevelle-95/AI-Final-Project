"""
CLIP-CA-CG — FastAPI Backend Server
Multimodal Sentiment Analysis with RoBERTa + ResNet-50

Run:
    python -m uvicorn server:app --reload --port 8000
"""

import io
import os
import re
import csv
import time
import uuid
import yaml
import base64
import shutil
import traceback
from pathlib import Path
from typing import Optional, List

import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import RobertaTokenizer

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from models.model import CLIPCACG
from datasets.multimodal_dataset import multimodal_collate_fn

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "configs" / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SERVER_START_TIME = time.time()

# ──────────────────────────────────────────────────────────────
# Global State
# ──────────────────────────────────────────────────────────────
model: Optional[CLIPCACG] = None
tokenizer: Optional[RobertaTokenizer] = None
image_transform = None
model_loaded = False
checkpoint_path: Optional[str] = None
is_trained = False  # True only if a real checkpoint was loaded

# In-memory results store  {review_id: {...}}
results_store: dict = {}
result_counter = 0


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def find_latest_checkpoint(checkpoint_dir: Path):
    """Find the most recent model_epoch_*.pt file."""
    files = list(checkpoint_dir.glob("model_epoch_*.pt"))
    if not files:
        return None

    def get_epoch(p):
        m = re.findall(r"\d+", p.stem)
        return int(m[-1]) if m else -1

    return max(files, key=get_epoch)


def load_model():
    """Load the CLIP-CA-CG model and tokenizer."""
    global model, tokenizer, image_transform, model_loaded, checkpoint_path, is_trained

    print("[server] Loading RoBERTa tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(CONFIG["model"]["text_model"])
    print(f"[server] Tokenizer loaded — {tokenizer.vocab_size} tokens")

    image_size = CONFIG["data"]["image_size"]
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    print("[server] Initializing CLIP-CA-CG model...")
    model = CLIPCACG(
        num_classes=CONFIG["model"].get("num_classes", 3),
        dropout=CONFIG["model"].get("dropout", 0.3)
    ).to(DEVICE)

    # Try loading checkpoint
    cp_dir = BASE_DIR / CONFIG["output"]["checkpoint_dir"]
    cp = find_latest_checkpoint(cp_dir)

    if cp is not None:
        print(f"[server] Loading checkpoint: {cp}")
        state = torch.load(cp, map_location=torch.device(DEVICE))
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        checkpoint_path = str(cp)
        is_trained = True
        print("[server] Checkpoint loaded successfully")
    else:
        checkpoint_path = None
        is_trained = False
        print("[server] WARNING: No checkpoint found — using random weights")

    model.eval()
    model_loaded = True
    print(f"[server] Model ready on {DEVICE}")


def preprocess_text(text: str, max_length: int = None):
    """Tokenize text using RoBERTa tokenizer."""
    if max_length is None:
        max_length = CONFIG["data"]["max_length"]

    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return enc["input_ids"], enc["attention_mask"]


def preprocess_image(img: Image.Image):
    """Transform a PIL Image for the model."""
    img = img.convert("RGB")
    return image_transform(img)


def predict_single(text: str, image: Optional[Image.Image] = None):
    """
    Run a single review through the full CLIP-CA-CG pipeline.
    Returns prediction dict with sentiment, confidence, probabilities.
    """
    if not model_loaded:
        raise RuntimeError("Model not loaded")

    # 1. Text
    input_ids, attention_mask = preprocess_text(text)
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)

    # 2. Image
    if image is not None:
        img_tensor = preprocess_image(image).unsqueeze(0)  # (1, C, H, W)
    else:
        # Blank image when none provided
        blank = Image.new("RGB", (224, 224), (0, 0, 0))
        img_tensor = preprocess_image(blank).unsqueeze(0)

    # Shape: (B=1, N=1, C, H, W)
    images = img_tensor.unsqueeze(0).to(DEVICE)
    # Image mask: (B=1, N=1, 1)
    image_mask = torch.ones((1, 1, 1), dtype=torch.float32).to(DEVICE)
    if image is None:
        image_mask = torch.zeros((1, 1, 1), dtype=torch.float32).to(DEVICE)

    # 3. Forward pass
    start = time.time()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_mask=image_mask
        )
    inference_ms = (time.time() - start) * 1000

    logits = outputs.logits if hasattr(outputs, "logits") else outputs
    if isinstance(logits, tuple):
        logits = logits[0]

    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(probs.argmax())
    sentiment = LABEL_MAP[pred_idx]
    confidence = float(probs[pred_idx])

    # System action mapping
    action_map = {
        "positive": "Highlighted / Approved",
        "neutral": "Logged",
        "negative": "Flagged for Support"
    }

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4),
        "probabilities": {
            "negative": round(float(probs[0]), 4),
            "neutral": round(float(probs[1]), 4),
            "positive": round(float(probs[2]), 4),
        },
        "system_action": action_map[sentiment],
        "has_image": image is not None,
        "inference_time_ms": round(inference_ms, 1),
        "model_trained": is_trained,
    }


# ──────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="CLIP-CA-CG API",
    description="Multimodal Sentiment Analysis Backend",
    version="1.0.0",
)

# CORS — allow frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Load model on server startup."""
    try:
        load_model()
    except Exception as e:
        print(f"[server] ERROR loading model: {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    """Health check — returns model/GPU/uptime status."""
    uptime = time.time() - SERVER_START_TIME
    hours = int(uptime // 3600)
    mins = int((uptime % 3600) // 60)

    gpu_name = "N/A (CPU mode)"
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"

    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_trained": is_trained,
        "checkpoint": checkpoint_path,
        "device": DEVICE,
        "gpu": gpu_name,
        "gpu_memory": gpu_mem,
        "cuda_available": torch.cuda.is_available(),
        "text_encoder": CONFIG["model"]["text_model"],
        "image_encoder": CONFIG["model"]["image_model"],
        "num_classes": CONFIG["model"].get("num_classes", 3),
        "uptime": f"{hours}h {mins}m",
        "uptime_seconds": round(uptime),
        "total_results": len(results_store),
    }


@app.post("/api/predict")
async def predict(
    text: str = Form(...),
    product_title: str = Form(""),
    product_description: str = Form(""),
    stars: int = Form(0),
    image: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
):
    """
    Single review prediction via the CLIP-CA-CG pipeline.
    Accepts review text + optional image (file upload or base64).
    """
    global result_counter

    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Build full text (matching dataset format)
    full_text = f"{product_title} {product_description} {text}".strip()

    # Process image if provided
    pil_image = None
    if image is not None:
        try:
            img_bytes = await image.read()
            pil_image = Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    elif image_base64:
        try:
            # Strip data:image/...;base64, prefix if present
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]
            img_bytes = base64.b64decode(image_base64)
            pil_image = Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

    try:
        result = predict_single(full_text, pil_image)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # Store result
    result_counter += 1
    review_id = result_counter
    stored = {
        "id": review_id,
        "text": text,
        "product_title": product_title,
        "product_description": product_description,
        "stars": stars,
        **result,
    }
    results_store[review_id] = stored

    return {
        "review_id": review_id,
        **result,
    }


@app.post("/api/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file and run batch prediction on all rows.
    Required columns: product_title, review_text
    Optional: product_description, image_paths, label
    """
    global result_counter

    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Read CSV
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot parse CSV: {e}")

    # Validate columns
    required = ["product_title", "review_text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}. Found: {list(df.columns)}"
        )

    img_dir = BASE_DIR / "data" / "images"
    batch_results = []
    start_time = time.time()

    for idx, row in df.iterrows():
        # Build text
        title = str(row.get("product_title", ""))
        desc = str(row.get("product_description", ""))
        review = str(row.get("review_text", ""))
        full_text = f"{title} {desc} {review}".strip()

        # Try to load image(s)
        pil_image = None
        if "image_paths" in df.columns:
            img_paths_str = str(row["image_paths"])
            if img_paths_str and img_paths_str.lower() not in {"nan", "none", ""}:
                first_img = img_paths_str.split(";")[0].strip().strip('"')
                img_path = img_dir / first_img
                if img_path.exists():
                    try:
                        pil_image = Image.open(img_path)
                    except Exception:
                        pass

        try:
            pred = predict_single(full_text, pil_image)
        except Exception:
            pred = {
                "sentiment": "neutral",
                "confidence": 0.0,
                "probabilities": {"negative": 0.33, "neutral": 0.34, "positive": 0.33},
                "system_action": "Logged",
                "has_image": pil_image is not None,
                "inference_time_ms": 0,
                "model_trained": is_trained,
            }

        result_counter += 1
        review_id = result_counter
        stars = int(row.get("stars", row.get("rating", 0))) if "stars" in df.columns or "rating" in df.columns else 0
        label = int(row["label"]) if "label" in df.columns else None

        stored = {
            "id": review_id,
            "text": review,
            "product_title": title,
            "product_description": desc,
            "stars": stars,
            "ground_truth_label": label,
            **pred,
        }
        results_store[review_id] = stored
        batch_results.append(stored)

    total_time = time.time() - start_time

    # Compute summary stats
    sentiments = [r["sentiment"] for r in batch_results]
    pos_count = sentiments.count("positive")
    neu_count = sentiments.count("neutral")
    neg_count = sentiments.count("negative")
    total = len(batch_results)
    avg_conf = sum(r["confidence"] for r in batch_results) / max(total, 1)

    return {
        "status": "success",
        "file": file.filename,
        "total_reviews": total,
        "processing_time_ms": round(total_time * 1000),
        "summary": {
            "positive": pos_count,
            "neutral": neu_count,
            "negative": neg_count,
            "positive_pct": round(pos_count / max(total, 1) * 100, 1),
            "neutral_pct": round(neu_count / max(total, 1) * 100, 1),
            "negative_pct": round(neg_count / max(total, 1) * 100, 1),
            "avg_confidence": round(avg_conf, 4),
        },
        "results": batch_results,
        "model_trained": is_trained,
        "columns_found": list(df.columns),
    }


@app.post("/api/upload/images")
async def upload_images(files: List[UploadFile] = File(...)):
    """Upload review images to data/images/ directory."""
    img_dir = BASE_DIR / "data" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        try:
            content = await f.read()
            dest = img_dir / f.filename
            with open(dest, "wb") as fh:
                fh.write(content)
            saved.append(f.filename)
        except Exception as e:
            saved.append(f"ERROR: {f.filename} — {e}")

    return {
        "status": "success",
        "uploaded": len(saved),
        "files": saved,
    }


@app.get("/api/results")
async def get_results(sentiment: Optional[str] = None):
    """
    Get all stored prediction results.
    Optional filter by sentiment: positive, neutral, negative.
    """
    items = list(results_store.values())

    if sentiment and sentiment in LABEL_MAP.values():
        items = [r for r in items if r["sentiment"] == sentiment]

    # Summary stats
    total = len(list(results_store.values()))
    all_items = list(results_store.values())
    pos = sum(1 for r in all_items if r["sentiment"] == "positive")
    neu = sum(1 for r in all_items if r["sentiment"] == "neutral")
    neg = sum(1 for r in all_items if r["sentiment"] == "negative")
    avg_conf = sum(r["confidence"] for r in all_items) / max(total, 1) if total else 0

    return {
        "total": total,
        "filtered": len(items),
        "summary": {
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "positive_pct": round(pos / max(total, 1) * 100, 1),
            "neutral_pct": round(neu / max(total, 1) * 100, 1),
            "negative_pct": round(neg / max(total, 1) * 100, 1),
            "avg_confidence": round(avg_conf, 4),
        },
        "results": items,
        "model_trained": is_trained,
    }


@app.get("/api/results/{review_id}")
async def get_result(review_id: int):
    """Get prediction result for a specific review."""
    if review_id not in results_store:
        raise HTTPException(status_code=404, detail=f"Review {review_id} not found")
    return results_store[review_id]


@app.get("/api/export/csv")
async def export_csv():
    """Export all results as a downloadable CSV file."""
    if not results_store:
        raise HTTPException(status_code=404, detail="No results to export")

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "review_id", "product_title", "review_text", "stars",
        "predicted_sentiment", "confidence",
        "prob_negative", "prob_neutral", "prob_positive",
        "system_action", "has_image", "model_trained",
    ])

    for r in results_store.values():
        probs = r.get("probabilities", {})
        writer.writerow([
            r["id"],
            r.get("product_title", ""),
            r.get("text", ""),
            r.get("stars", ""),
            r["sentiment"],
            r["confidence"],
            probs.get("negative", ""),
            probs.get("neutral", ""),
            probs.get("positive", ""),
            r["system_action"],
            r.get("has_image", False),
            r.get("model_trained", False),
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=clip_cacg_predictions.csv"}
    )


@app.delete("/api/results")
async def clear_results():
    """Clear all stored results."""
    global result_counter
    results_store.clear()
    result_counter = 0
    return {"status": "cleared", "total": 0}
