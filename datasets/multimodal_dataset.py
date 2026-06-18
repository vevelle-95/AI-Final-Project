import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class MultiModalDataset(Dataset):
    def __init__(self, df, tokenizer, image_transform, img_dir):
        self.df = df
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def load_images(self, image_list):
        images = []
        base = Path(self.img_dir)

        for img_name in image_list:
            # sanitize filename from CSV
            img_name = str(img_name).strip()
            img_name = img_name.strip('"').strip("'")
            if img_name == "" or img_name.lower() in {"nan", "none"}:
                continue

            path = base / img_name

            #print("Trying image path:", repr(path))
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            img = Image.open(path).convert("RGB")
            img = self.image_transform(img)
            images.append(img)

        if len(images) == 0:
            raise RuntimeError("No valid images found for this sample.")
        return torch.stack(images)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---- TEXT MERGE ----
        text = (
            str(row["product_title"]) + " " +
            str(row["product_description"]) + " " +
            str(row["review_text"])
        )

        text_enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        # ---- IMAGE PROCESSING ----
        image_files = str(row["image_paths"]).split(";")
        images = self.load_images(image_files)

        return {
            "input_ids": text_enc["input_ids"].squeeze(0),
            "attention_mask": text_enc["attention_mask"].squeeze(0),
            "images": images,  # multiple images tensor
            "label": torch.tensor(row["label"], dtype=torch.long)
        }

def multimodal_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    images_list = [item["images"] for item in batch]
    
    # Get the TRUE maximum number of loaded images in this specific batch
    max_n = max(img.shape[0] for img in images_list)

    padded_images = []
    binary_masks = []

    # Loop through every sample to pad images and generate a matching mask
    for imgs in images_list:
        n, c, h, w = imgs.shape
        
        mask = torch.zeros((max_n, 1), dtype=torch.float32)
        mask[:n, :] = 1.0
        binary_masks.append(mask)

        if n < max_n:
            pad = torch.zeros((max_n - n, c, h, w), dtype=imgs.dtype, device=imgs.device)
            imgs = torch.cat([imgs, pad], dim=0)
            
        padded_images.append(imgs)

    images = torch.stack(padded_images)       
    image_mask = torch.stack(binary_masks)    

    # Create the base output dictionary
    output_batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": images,           
        "image_mask": image_mask
    }

    # 🔥 THE SMART FIX: Only try to stack labels if they actually exist in the batch!
    if "label" in batch[0]:
        output_batch["label"] = torch.stack([item["label"] for item in batch])

    return output_batch