import yaml
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from torchvision import transforms
import argparse
import re

from datasets.multimodal_dataset import MultiModalDataset, multimodal_collate_fn
from models.model import CLIPCACG
from training.eval import evaluate

BASE_DIR = Path(__file__).resolve().parent


# -------------------------
# FIND LATEST CHECKPOINT
# -------------------------
def find_latest_checkpoint(checkpoint_dir: Path):
    files = list(checkpoint_dir.glob("model_epoch_*.pt"))

    if not files:
        return None

    def get_epoch(path):
        match = re.findall(r"\d+", path.stem)
        return int(match[-1]) if match else -1

    return max(files, key=get_epoch)


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to evaluation CSV")
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Path to checkpoint OR 'latest'"
    )
    args = parser.parse_args()

    # Load config
    with open(BASE_DIR / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)

    # Load data
    df = pd.read_csv(args.csv)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config["model"]["text_model"])

    # Image transform
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

    dataset = MultiModalDataset(df, tokenizer, transform, img_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=multimodal_collate_fn
    )

    # -------------------------
    # MODEL
    # -------------------------
    model = CLIPCACG(num_classes=3).to(device)

    # -------------------------
    # CHECKPOINT LOADING (FIXED)
    # -------------------------
    if args.checkpoint is None or args.checkpoint == "latest":
        cp = find_latest_checkpoint(BASE_DIR / "outputs" / "checkpoints")

        if cp is None:
            raise FileNotFoundError("No checkpoints found")

        print(f"[INFO] Auto-loading latest checkpoint: {cp}")

    else:
        cp = Path(args.checkpoint)
        if not cp.exists():
            raise FileNotFoundError(f"Checkpoint not found: {cp}")

    print(f"[INFO] Using checkpoint: {cp}")

    checkpoint = torch.load(cp, map_location=torch.device(device))
    
    # handle both save formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print("[INFO] Model loaded successfully")

    # -------------------------
    # EVALUATION
    # -------------------------
    evaluate(model, dataloader, device)