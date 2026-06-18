import torch.nn as nn


class Projection(nn.Module):
    def __init__(self, in_dim, out_dim=512, dropout=0.1):
        super().__init__()

        # Two-layer MLP: nonlinear mapping needed to align RoBERTa's semantic
        # space with ResNet-50's visual space. A single linear layer cannot
        # bend or warp the input manifold, causing misaligned cross-modal
        # attention scores and raising hallucination risk.
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),

            # FIX (epoch ≥ 20): Add a final LayerNorm on the output.
            # Without it, repeated gradient updates push the projected
            # embeddings to ever-larger magnitudes. By epoch 20+, the
            # cross-attention dot-products (Q·Kᵀ / √d) become numerically
            # huge, softmax collapses to a one-hot, and the model effectively
            # hallucinates cross-modal context it has stopped attending to.
            # This norm keeps the output on a stable hypersphere regardless
            # of how many epochs have elapsed.
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.mlp(x)