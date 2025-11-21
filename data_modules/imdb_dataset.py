from typing import Any, List, Tuple, Dict
import re
from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence
import datasets
from types import SimpleNamespace

from ._base import BaseDataset


class IMDBDataset(BaseDataset):
    def __init__(self, split: str, max_vocab_size: int = 50000, min_freq: int = 2):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        dataset = datasets.load_dataset("stanfordnlp/imdb", split="train" if split == "train" else "test")

        if split == "train":
            train_data = dataset
        else:
            train_data = datasets.load_dataset("stanfordnlp/imdb", split="train")

        self.vocab = self._build_vocab(train_data, max_vocab_size, min_freq)
        self.vocab_size = len(self.vocab)
        self.pad_idx = self.vocab["<pad>"]
        self.unk_idx = self.vocab["<unk>"]

        self.texts = []
        self.labels = []
        for item in dataset:
            tokens = self._tokenize(item["text"])
            token_ids = [self.vocab.get(token, self.unk_idx) for token in tokens]
            self.texts.append(torch.tensor(token_ids, dtype=torch.long))
            self.labels.append(item["label"])

        self.inferred_params = SimpleNamespace(
            vocab_size=self.vocab_size,
            num_classes=2,
        )

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    def _build_vocab(self, dataset, max_vocab_size: int, min_freq: int) -> Dict[str, int]:
        counter = Counter()
        for item in dataset:
            tokens = self._tokenize(item["text"])
            counter.update(tokens)

        vocab = {"<pad>": 0, "<unk>": 1}

        for word, freq in counter.most_common(max_vocab_size - len(vocab)):
            if freq >= min_freq:
                vocab[word] = len(vocab)

        return vocab

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.texts[index], self.labels[index]

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        texts = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        lengths = torch.tensor([len(text) for text in texts], dtype=torch.long)

        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)

        return padded_texts, lengths, labels


