import os
import random
from pathlib import Path
from typing import Tuple, List

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10


# -----------------------------------------------------------------------------
# Synthetic fallback datasets (used when remote download fails)
# -----------------------------------------------------------------------------


class SyntheticImageDataset(Dataset):
    def __init__(self, num_samples: int = 1000, num_classes: int = 10, image_size: int = 32):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.data = torch.randn(num_samples, 3, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SyntheticTextDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 1000,
        seq_len: int = 32,
        vocab_size: int = 5000,
        num_classes: int = 2,
    ):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.data = torch.randint(1, vocab_size, (num_samples, seq_len))
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# -----------------------------------------------------------------------------
# Text tokeniser & vocabulary builder (simple whitespace split)
# -----------------------------------------------------------------------------


def build_vocab(corpus: List[str], max_size: int = 30000):
    from collections import Counter

    counter = Counter()
    for line in corpus:
        counter.update(line.split())
    most_common = counter.most_common(max_size - 2)  # reserve PAD & UNK
    vocab = {w: i + 2 for i, (w, _) in enumerate(most_common)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1
    return vocab


def encode_text(text: str, vocab: dict, max_length: int):
    tokens = text.split()
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens[:max_length]]
    if len(ids) < max_length:
        ids.extend([vocab["<PAD>"]] * (max_length - len(ids)))
    return torch.tensor(ids)


# -----------------------------------------------------------------------------
# Dataloader builder (public API used by train.py)
# -----------------------------------------------------------------------------


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, int]:
    if cfg.task == "image_classification":
        return _build_image_dataloaders(cfg)
    elif cfg.task == "text_classification":
        return _build_text_dataloaders(cfg)
    else:
        raise ValueError(f"Unknown task {cfg.task}")


# -----------------------------------------------------------------------------
# Image pipeline
# -----------------------------------------------------------------------------

def _image_transforms(cfg):
    train_tf = []
    if cfg.dataset.augmentation.random_crop:
        train_tf.append(T.RandomCrop(cfg.dataset.image_size, padding=4))
    if cfg.dataset.augmentation.horizontal_flip:
        train_tf.append(T.RandomHorizontalFlip())
    train_tf.append(T.ToTensor())
    train_tf.append(T.Normalize(mean=cfg.dataset.normalization.mean, std=cfg.dataset.normalization.std))

    val_tf = T.Compose(
        [T.ToTensor(), T.Normalize(mean=cfg.dataset.normalization.mean, std=cfg.dataset.normalization.std)]
    )
    return T.Compose(train_tf), val_tf


def _build_image_dataloaders(cfg):
    # Attempt to fetch CIFAR-10; fallback to synthetic
    train_tf, val_tf = _image_transforms(cfg)
    num_classes = 10
    try:
        train_set_full = CIFAR10(
            root=Path("~/.cache/datasets").expanduser(),
            train=True,
            transform=train_tf,
            download=True,
        )
        val_set_full = CIFAR10(
            root=Path("~/.cache/datasets").expanduser(),
            train=True,
            transform=val_tf,
            download=True,
        )
        # Manual split
        total_size = len(train_set_full)
        split_point = int(total_size * cfg.dataset.train_val_split[0])
        idx = torch.randperm(total_size)
        train_idx, val_idx = idx[:split_point], idx[split_point:]
        train_set = torch.utils.data.Subset(train_set_full, train_idx)
        val_set = torch.utils.data.Subset(val_set_full, val_idx)
    except Exception as e:
        print(f"Dataset download failed ({e}); using synthetic CIFAR-10 data.")
        train_set = SyntheticImageDataset(5000, num_classes, cfg.dataset.image_size)
        val_set = SyntheticImageDataset(1000, num_classes, cfg.dataset.image_size)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )
    return train_loader, val_loader, num_classes


# -----------------------------------------------------------------------------
# Text pipeline
# -----------------------------------------------------------------------------

# For the sake of resource limitation, we only implement synthetic fallback.


def _build_text_dataloaders(cfg):
    # Attempt to load dataset from TSV or JSONL if available locally.
    root_path = Path("datasets") / cfg.dataset.name
    text_samples = []
    label_samples = []
    if root_path.exists():
        for line in root_path.open():
            try:
                record = json.loads(line)
                text_samples.append(record[cfg.dataset.text_field])
                label_samples.append(record[cfg.dataset.label_field])
            except Exception:
                continue
    else:
        print("Text dataset not found locally; using synthetic data.")
        num_classes = 2
        seq_len = cfg.dataset.max_length
        train_set = SyntheticTextDataset(2000, seq_len, 5000, num_classes)
        val_set = SyntheticTextDataset(500, seq_len, 5000, num_classes)
        train_loader = DataLoader(train_set, batch_size=cfg.dataset.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=cfg.dataset.batch_size, shuffle=False)
        return train_loader, val_loader, num_classes

    # Build vocabulary
    vocab = build_vocab(text_samples, max_size=cfg.model.tokenizer.vocab_size)
    num_classes = len(set(label_samples))

    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.enc = [encode_text(t, vocab, cfg.dataset.max_length) for t in texts]
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.enc[idx], self.labels[idx]

    total_size = len(text_samples)
    idx = list(range(total_size))
    random.shuffle(idx)
    split_point = int(total_size * cfg.dataset.train_val_split[0])
    train_idx, val_idx = idx[:split_point], idx[split_point:]
    train_texts = [text_samples[i] for i in train_idx]
    val_texts = [text_samples[i] for i in val_idx]
    train_labels = [label_samples[i] for i in train_idx]
    val_labels = [label_samples[i] for i in val_idx]

    train_set = TextDataset(train_texts, train_labels)
    val_set = TextDataset(val_texts, val_labels)
    train_loader = DataLoader(train_set, batch_size=cfg.dataset.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.dataset.batch_size, shuffle=False)
    return train_loader, val_loader, num_classes
