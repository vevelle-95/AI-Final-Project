import torch.nn as nn
from transformers import RobertaModel

class TextEncoder(nn.Module):
    def __init__(self, hidden_dim=768, bigru_hidden=384):
        super().__init__()
        self.model = RobertaModel.from_pretrained("roberta-base")
        self.bigru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=bigru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use full sequence output for Bi-GRU instead of just pooler_output
        sequence_output = out.last_hidden_state  # (B, seq_len, 768)
        gru_out, _ = self.bigru(sequence_output)  # (B, seq_len, 768)
        # Take the last time step as the sentence representation
        return gru_out[:, -1, :]  # (B, 768)
