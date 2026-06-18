import torch
import torch.nn as nn


class Gating(nn.Module):
    def __init__(self, dim=512, dropout=0.1):
        super().__init__()
        self.fc      = nn.Linear(dim * 2, dim)
        self.norm    = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, image):
        # Guard against 3-D tensors arriving from CrossAttention.
        # torch.cat(..., dim=1) on [B, S, D] inputs concatenates along the
        # sequence axis instead of the feature axis — silently wrong.
        # Mean-pool to [B, D] first.
        if text.dim() == 3:
            text  = text.mean(dim=1)   # [B, S, D] → [B, D]
        if image.dim() == 3:
            image = image.mean(dim=1)  # [B, S, D] → [B, D]

        combined = torch.cat([text, image], dim=-1)   # [B, D*2]

        # FIX (epoch ≥ 20): Clamp the gate away from the extremes [0, 1].
        #
        # Without clamping, the fc weights grow with each gradient update.
        # By epoch 20+ the pre-sigmoid activations are so large that sigmoid
        # returns values indistinguishable from 0.0 or 1.0 — a hard switch
        # that discards one modality entirely.  The fused representation then
        # contains only text OR only image information; the dropped modality
        # is effectively hallucinated rather than integrated.
        #
        # Clamping to [0.05, 0.95] keeps the gate a true soft blend at every
        # epoch, ensuring both modalities always contribute to the output.
        g = torch.sigmoid(self.fc(combined)).clamp(0.05, 0.95)   # [B, D]

        fused = g * text + (1 - g) * image                        # [B, D]
        return self.norm(self.dropout(fused))