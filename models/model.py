import torch
import torch.nn as nn

from models.encoders.text_encoder import TextEncoder
from models.encoders.image_encoder import ImageEncoder
from models.fusion.projection import Projection
from models.fusion.cross_attention import CrossAttention
from models.fusion.gating import Gating

class CLIPCACG(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()

        self.text_encoder = TextEncoder()
        self.image_encoder = ImageEncoder()

        self.text_proj = Projection(768)
        self.image_proj = Projection(2048)

        self.cross_attn = CrossAttention()
        self.gate = Gating()

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(512, num_classes)

    def freeze_backbones(self):
        for param in self.text_encoder.model.parameters():
            param.requires_grad = False
        for param in self.image_encoder.backbone.parameters():
            param.requires_grad = False
        print("[INFO] Backbones frozen — training projection, attention, gating, classifier only")

    def unfreeze_backbones(self):
        for param in self.text_encoder.model.parameters():
            param.requires_grad = True
        for param in self.image_encoder.backbone.parameters():
            param.requires_grad = True
        print("[INFO] Backbones unfrozen — fine-tuning entire model")

    def forward(self, input_ids, attention_mask, images, image_mask):
        # -----------------
        # 1. TEXT PROCESSING
        # -----------------
        t = self.text_encoder(input_ids, attention_mask)
        t = self.text_proj(t)

        # ------------------
        # 2. IMAGE PROCESSING
        # ------------------
        B, N, C, H, W = images.shape

        i = self.image_encoder(images.view(B * N, C, H, W))
        i = i.view(B, N, -1)       # (B, N, 2048)
        i = self.image_proj(i)     # (B, N, 512)

        # -----------------------
        # 3. CROSS ATTENTION & FUSION
        # -----------------------
        t, i = self.cross_attn(t, i)

        mask = image_mask.to(i.device)  # (B, N, 1)
        i = i * mask
        i = i.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, 512)

        # GATING, DROPOUT & CLASSIFIER
        f = self.gate(t, i)
        f = self.dropout(f)
        out = self.classifier(f)

        return out
