import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, dim=512, heads=4, dropout=0.1):
        super().__init__()

        # Separate attention modules for each direction.
        # Sharing a single nn.MultiheadAttention for T→I and I→T forces the
        # same Q/K/V weights to serve opposite roles, causing underdetermined
        # cross-modal alignment and raising hallucination risk.
        self.text_to_image_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.image_to_text_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )

        # Pre-norm before each attention block.
        # Without normalisation, attention logits grow unbounded and softmax
        # collapses to a one-hot, ignoring most cross-modal context.
        self.norm_text  = nn.LayerNorm(dim)
        self.norm_image = nn.LayerNorm(dim)

        # FIX (epoch ≥ 20): Post-residual LayerNorms.
        #
        # Each gradient update incrementally sharpens attention weight
        # distributions.  By epoch 20+ the attention entropy is so low that
        # the model concentrates almost entirely on 1–2 positions and ignores
        # the rest of the cross-modal context — effectively hallucinating the
        # portions it no longer attends to.
        #
        # Adding a LayerNorm *after* the residual connection re-centres and
        # re-scales the fused representation at every forward pass, breaking
        # the compounding feedback loop that drives attention collapse.
        # This mirrors the standard pre-norm + post-norm double-norm pattern
        # used in stable long-training transformers.
        self.norm_t_out = nn.LayerNorm(dim)
        self.norm_i_out = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, image):
        # --- ensure 3-D sequence tensors [B, S, D] ---
        squeezed_text  = text.dim()  == 2
        squeezed_image = image.dim() == 2

        if squeezed_text:
            text  = text.unsqueeze(1)   # [B, D] → [B, 1, D]
        if squeezed_image:
            image = image.unsqueeze(1)  # [B, D] → [B, 1, D]

        # --- pre-norm ---
        text_normed  = self.norm_text(text)
        image_normed = self.norm_image(image)

        # --- cross-attention (query, key, value) ---
        t_attn, _ = self.text_to_image_attn(
            query=text_normed, key=image_normed, value=image_normed
        )
        i_attn, _ = self.image_to_text_attn(
            query=image_normed, key=text_normed, value=text_normed
        )

        # Residual connections + post-residual norm (epoch ≥ 20 fix).
        # The original code had residuals but no output norm, so the
        # representation magnitude still drifted upward with each epoch,
        # feeding back into the pre-norm and amplifying attention sharpening.
        t_out = self.norm_t_out(text  + self.dropout(t_attn))   # [B, S_text,  D]
        i_out = self.norm_i_out(image + self.dropout(i_attn))   # [B, S_image, D]

        # --- restore 2-D if input was 2-D ---
        if squeezed_text:
            t_out = t_out.squeeze(1)   # [B, 1, D] → [B, D]
        if squeezed_image:
            i_out = i_out.squeeze(1)   # [B, 1, D] → [B, D]

        return t_out, i_out