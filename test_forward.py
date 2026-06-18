import torch
import pandas as pd
from transformers import RobertaTokenizer
from torchvision import transforms
from pathlib import Path

from datasets.multimodal_dataset import MultiModalDataset
from models.model import CLIPCACG

# -------------------------
# LOAD CSV (script-relative path)
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "data" / "test.csv")

# -------------------------
# TOKENIZER
# -------------------------
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# -------------------------
# IMAGE TRANSFORM
# -------------------------
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------
# DATASET
# -------------------------
dataset = MultiModalDataset(
    df=df,
    tokenizer=tokenizer,
    image_transform=image_transform,
    img_dir=str(BASE_DIR / "data" / "images")
)

# -------------------------
# MODEL
# -------------------------
model = CLIPCACG(num_classes=3)
model.eval()

# -------------------------
# GET ONE SAMPLE
# -------------------------
sample = dataset[0]

input_ids = sample["input_ids"].unsqueeze(0)
attention_mask = sample["attention_mask"].unsqueeze(0)
images = sample["images"].unsqueeze(0)
image_mask = torch.ones(1, images.shape[1])  # all images are real

# -------------------------
# FORWARD PASS TEST
# -------------------------
with torch.no_grad():
    output = model(input_ids, attention_mask, images, image_mask)

print("Output shape:", output.shape)
print("Output logits:", output)