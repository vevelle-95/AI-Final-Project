import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize the gradient scaler for mixed precision training
        is_cuda = "cuda" in str(self.device)
        self.scaler = GradScaler(device="cuda" if is_cuda else None, enabled=is_cuda)

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            # 1. Move data to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            images = batch["images"].to(self.device)
            image_mask = batch["image_mask"].to(self.device)

            self.optimizer.zero_grad()

            # 2. Forward pass under autocast (runs fast, uses less VRAM)
            # Dynamic device type matching (cuda vs cpu)
            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            with autocast(device_type=device_type, enabled=(device_type == "cuda")):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    image_mask=image_mask
                )
                
                # Safety check: Extract logits if your model returns a dict/tuple
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                if isinstance(logits, tuple):
                    logits = logits[0]

                loss = self.criterion(logits, labels)

            # 3. Backward pass using the scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(dataloader)