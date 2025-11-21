import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from typing import Optional


class LSTMModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        pad_idx: int = 0,
        max_seq_len: Optional[int] = None,
        embedding_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.vocab_size = int(vocab_size)
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.pad_idx = int(pad_idx)
        self.max_seq_len = int(max_seq_len) if max_seq_len is not None else None
        self.num_directions = 2 if self.bidirectional else 1

        lstm_dropout = float(dropout) if self.num_layers > 1 else 0.0

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.pad_idx)
        self.embedding_dropout = nn.Dropout(float(embedding_dropout))
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.feature_dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(self.hidden_size * self.num_directions, self.num_classes)

    def _prepare_lengths(self, input_tensor: torch.Tensor, lengths: Optional[torch.Tensor]) -> torch.Tensor:
        if lengths is None:
            lengths = torch.full(
                (input_tensor.size(0),),
                input_tensor.size(1),
                dtype=torch.long,
                device=input_tensor.device,
            )
        else:
            lengths = torch.as_tensor(lengths, dtype=torch.long, device=input_tensor.device)

        seq_len = input_tensor.size(1)
        lengths = torch.clamp(lengths, min=1, max=seq_len)
        return lengths

    def forward(self, input_ids, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        dev = self.embedding.weight.device
        input_tensor = torch.as_tensor(input_ids, dtype=torch.long, device=dev)

        if self.max_seq_len is not None and input_tensor.size(1) > self.max_seq_len:
            input_tensor = input_tensor[:, : self.max_seq_len]
            if lengths is not None:
                lengths = torch.clamp(torch.as_tensor(lengths, dtype=torch.long), max=self.max_seq_len)

        lengths_tensor = self._prepare_lengths(input_tensor, lengths)

        embeddings = self.embedding(input_tensor)
        embeddings = self.embedding_dropout(embeddings)

        packed = pack_padded_sequence(
            embeddings,
            lengths_tensor.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)

        hidden = hidden.view(self.num_layers, self.num_directions, input_tensor.size(0), self.hidden_size)
        last_layer = hidden[-1]
        if self.bidirectional:
            features = torch.cat((last_layer[0], last_layer[1]), dim=-1)
        else:
            features = last_layer[0]

        features = self.feature_dropout(features)
        logits = self.classifier(features)
        return logits
