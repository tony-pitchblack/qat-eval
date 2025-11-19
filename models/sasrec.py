import torch
import torch.nn as nn
from typing import Optional, Union


def _generate_subsequent_mask(seq_len: int, device: Union[str, torch.device]) -> torch.Tensor:
    mask = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
    return mask


class SASRecModel(nn.Module):
    def __init__(
        self,
            num_items: int,
            max_seq_len: int,
            hidden_size: int = 64,
            num_heads: int = 1,
            num_layers: int = 1,
            dropout: float = 0.1,
            pad_token_id: int = 0,
            device: Optional[Union[str, torch.device]] = None,
        ):
        super().__init__()
        self.num_items = int(num_items)
        self.max_seq_len = int(max_seq_len)
        self.hidden_size = int(hidden_size)
        self.pad_token_id = int(pad_token_id)
        self.device_ref = torch.device(device) if isinstance(device, str) else device

        self.item_embedding = nn.Embedding(self.num_items + 1, self.hidden_size)
        self.positional_embedding = nn.Embedding(self.max_seq_len, self.hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.final_norm = nn.LayerNorm(self.hidden_size, eps=1e-8)
        self.output = nn.Linear(self.hidden_size, self.num_items)

    def forward(self, item_sequence, sequence_lengths):
        dev = self.device_ref if self.device_ref is not None else (item_sequence.device if isinstance(item_sequence, torch.Tensor) else "cpu")
        items_t = torch.as_tensor(item_sequence, device=dev, dtype=torch.long)
        lens_t = torch.as_tensor(sequence_lengths, device=dev, dtype=torch.long)

        batch_size, seq_len = items_t.shape
        seq_len = min(seq_len, self.max_seq_len)
        if items_t.shape[1] != seq_len:
            items_t = items_t[:, :seq_len]
            lens_t = torch.clamp(lens_t, max=seq_len)

        x = self.item_embedding(items_t) * (self.hidden_size ** 0.5)
        positions = torch.arange(seq_len, device=dev, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        x = x + self.positional_embedding(positions)
        x = self.embedding_dropout(x)

        pad_mask = (items_t == self.pad_token_id) | (items_t == self.num_items)
        attn_mask = _generate_subsequent_mask(seq_len, device=dev)

        x = self.encoder(x, mask=attn_mask, src_key_padding_mask=pad_mask)
        x = self.final_norm(x)

        last_index = torch.clamp(lens_t - 1, min=0)
        batch_idx = torch.arange(batch_size, device=dev)
        gathered = x[batch_idx, last_index]

        logits = self.output(gathered)
        return logits


