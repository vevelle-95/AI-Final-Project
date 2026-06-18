import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from torchvision import transforms

from datasets.multimodal_dataset import MultiModalDataset, multimodal_collate_fn
from models.model import CLIPCACG
from training.engine import Trainer

def main(df, config, base_dir):
    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Text preprocessing setup
    tokenizer = RobertaTokenizer.from_pretrained(config["model"]["text_model"])

    # 3. Vision preprocessing setup
    image_size = config["data"]["image_size"]
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    # 4. Data Pipeline
    img_dir = str(base_dir / "data" / "images")
    dataset = MultiModalDataset(df, tokenizer, transform, img_dir)

    use_cuda = device.type == "cuda"
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=multimodal_collate_fn,
        pin_memory=use_cuda,
        drop_last=True
    )

    print("Train dataset rows:", len(dataset))
    print("Train batches:", len(dataloader), " batch_size:", config["training"]["batch_size"])

    # 5. Model initialization
    dropout = config["model"].get("dropout", 0.3)
    num_classes = config["model"].get("num_classes", 3)
    model = CLIPCACG(num_classes=num_classes, dropout=dropout).to(device)

    checkpoint_dir = base_dir / config["output"]["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    phase1_epochs = config["training"].get("phase1_epochs", 10)
    phase2_epochs = config["training"].get("phase2_epochs", 10)
    lr = float(config["training"]["lr"])
    finetune_lr = float(config["training"].get("finetune_lr", lr * 0.1))

    # -------------------------
    # PHASE 1 — Frozen backbones
    # Train only projection, attention, gating, classifier
    # -------------------------
    model.freeze_backbones()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
    trainer = Trainer(model, optimizer, device)

    print(f"\n=== PHASE 1: Frozen backbones ({phase1_epochs} epochs) ===")
    for epoch in range(phase1_epochs):
        loss = trainer.train_one_epoch(dataloader)
        print(f"Epoch {epoch+1}/{phase1_epochs} | Avg Loss: {loss:.4f}")
        checkpoint_path = checkpoint_dir / f"model_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    # -------------------------
    # PHASE 2 — Unfrozen fine-tuning
    # Fine-tune entire model at lower learning rate
    # -------------------------
    model.unfreeze_backbones()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=finetune_lr, weight_decay=0.01)
    trainer = Trainer(model, optimizer, device)

    print(f"\n=== PHASE 2: Full fine-tuning ({phase2_epochs} epochs, lr={finetune_lr}) ===")
    for epoch in range(phase2_epochs):
        loss = trainer.train_one_epoch(dataloader)
        global_epoch = phase1_epochs + epoch + 1
        print(f"Epoch {global_epoch}/{phase1_epochs + phase2_epochs} | Avg Loss: {loss:.4f}")
        checkpoint_path = checkpoint_dir / f"model_epoch_{global_epoch}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
