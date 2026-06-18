import yaml
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer
from torchvision import transforms
from PIL import Image
import argparse
import re

from models.model import CLIPCACG
from datasets.multimodal_dataset import multimodal_collate_fn

BASE_DIR = Path(__file__).resolve().parent

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


# -------------------------
# FIND LATEST CHECKPOINT
# -------------------------
def find_latest_checkpoint(checkpoint_dir: Path):
    files = list(checkpoint_dir.glob("model_epoch_*.pt"))
    if not files:
        return None

    def get_epoch(p):
        m = re.findall(r"\d+", p.stem)
        return int(m[-1]) if m else -1

    return max(files, key=get_epoch)


# -------------------------
# DATASET
# -------------------------
class PredictionDataset(Dataset):
    def __init__(self, df, tokenizer, image_transform, img_dir, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.img_dir = img_dir
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def _load_images(self, image_list):
        images = []
        for img_name in image_list:
            path = f"{self.img_dir}/{img_name}"
            try:
                img = Image.open(path).convert("RGB")
                images.append(self.image_transform(img))
            except Exception as e:
                # Fallback to a blank image if one of the paths fails to load
                # to prevent the tensor batch tracking from crashing
                blank_img = Image.new("RGB", (224, 224), (0, 0, 0))
                images.append(self.image_transform(blank_img))
        
        # If no images were successfully loaded, provide at least 1 blank frame
        if not images:
            blank_img = Image.new("RGB", (224, 224), (0, 0, 0))
            images.append(self.image_transform(blank_img))
            
        return torch.stack(images)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = (
            str(row["product_title"]) + " " +
            str(row["product_description"]) + " " +
            str(row["review_text"])
        )

        text_enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,  # Dynamically read from configuration
            return_tensors="pt"
        )

        image_files = [
            f.strip().strip('"')
            for f in str(row["image_paths"]).split(";")
            if f.strip()
        ]

        return {
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "images": self._load_images(image_files),
        }


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    # config
    with open(BASE_DIR / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(args.csv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = RobertaTokenizer.from_pretrained(config["model"]["text_model"])

    image_size = config["data"]["image_size"]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    img_dir = str(BASE_DIR / "data" / "images")

    # Pass max_length cleanly out of config
    dataset = PredictionDataset(
        df, 
        tokenizer, 
        transform, 
        img_dir, 
        max_length=config["data"]["max_length"]
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,  # Explicitly 0 for Windows environment stability
        collate_fn=multimodal_collate_fn
    )

    # model
    model = CLIPCACG(num_classes=3).to(device)

    # -------------------------
    # CHECKPOINT LOGIC
    # -------------------------
    if args.checkpoint == "latest":
        cp = find_latest_checkpoint(BASE_DIR / "outputs" / "checkpoints")

        if cp is None:
            raise FileNotFoundError("No checkpoints found")

        print(f"[INFO] Loading latest checkpoint: {cp}")

    else:
        cp = Path(args.checkpoint)
        if not cp.exists():
            raise FileNotFoundError(f"Checkpoint not found: {cp}")

    # Enforce safe cross-backend tensor device mappings
    checkpoint = torch.load(cp, map_location=torch.device(device))
    model.load_state_dict(
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.eval()
    print(f"[INFO] Model loaded successfully: {cp}")

    # -------------------------
    # PREDICTION
    # -------------------------
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["images"].to(device)
            image_mask = batch["image_mask"].to(device)

            # Pass via exact named keyword mapping to avoid signature shuffling
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                image_mask=image_mask
            )

            # Safely trap logits out of structural class formats or raw tuples
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            if isinstance(logits, tuple):
                logits = logits[0]

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())   
            
    # -------------------------
    # SAVE OUTPUT
    # -------------------------
    df["predicted_label"] = all_preds
    df["predicted_sentiment"] = [LABEL_MAP[p] for p in all_preds]

    df["prob_negative"] = [round(float(p[0]), 4) for p in all_probs]
    df["prob_neutral"] = [round(float(p[1]), 4) for p in all_probs]
    df["prob_positive"] = [round(float(p[2]), 4) for p in all_probs]

    output_path = BASE_DIR / "outputs" / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n[INFO] Predictions saved to: {output_path}")
    print(df[["product_title", "predicted_sentiment"]].head().to_string())