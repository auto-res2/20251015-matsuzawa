"""Data preprocessing & DataLoader utilities."""
from pathlib import Path
from typing import Tuple
import json
import random

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from omegaconf import DictConfig

# ----------------------------------------------------------------------------
# Text dataset utilities
# ----------------------------------------------------------------------------

class AlpacaCharDataset(Dataset):
    """Character-level dataset for Alpaca-cleaned."""

    def __init__(self, file_path: Path, seq_length: int, char2idx: dict):
        self.samples = []
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text, label = obj["text"], obj["label"]
                self.samples.append((text.lower(), int(label)))
        self.seq_length = seq_length
        self.char2idx = char2idx
        self.pad_idx = self.char2idx["<pad>"]

    def __len__(self):
        return len(self.samples)

    def _encode(self, text: str):
        encoded = [self.char2idx.get(ch, self.char2idx["<unk>"]) for ch in text]
        encoded = encoded[: self.seq_length]
        if len(encoded) < self.seq_length:
            encoded += [self.pad_idx] * (self.seq_length - len(encoded))
        return torch.tensor(encoded, dtype=torch.long)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        return self._encode(text), torch.tensor(label, dtype=torch.long)


# ----------------------------------------------------------------------------
# Build DataLoaders
# ----------------------------------------------------------------------------

def build_dataloaders(cfg: DictConfig):
    if cfg.dataset.name.lower() == "cifar-10":
        return _build_cifar10(cfg)
    if cfg.dataset.name.lower() == "alpaca-cleaned":
        return _build_alpaca(cfg)
    raise ValueError(f"Unsupported dataset {cfg.dataset.name}")


def _build_cifar10(cfg: DictConfig):
    # Transforms
    normalize = T.Normalize(mean=cfg.dataset.normalization.mean, std=cfg.dataset.normalization.std)
    train_tfms = [
        T.RandomCrop(cfg.dataset.augmentation.random_crop.size, padding=cfg.dataset.augmentation.random_crop.padding),
        T.RandomHorizontalFlip(cfg.dataset.augmentation.random_horizontal_flip.p),
        T.ToTensor(),
        normalize,
    ]
    val_tfms = [T.ToTensor(), normalize]

    # Datasets (download if needed)
    path = Path(cfg.dataset.path)
    train_set = CIFAR10(root=path, train=True, transform=T.Compose(train_tfms), download=True)
    val_size = int((1 - cfg.dataset.split.train) * len(train_set))
    train_size = len(train_set) - val_size
    train_set, val_set = random_split(train_set, [train_size, val_size], generator=torch.Generator().manual_seed(cfg.training.seed))
    val_set.dataset.transform = T.Compose(val_tfms)  # type: ignore

    # A single batch for inference-time measurement
    sample_inputs = torch.randn(1, cfg.dataset.channels, cfg.dataset.input_size, cfg.dataset.input_size)

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, sample_inputs


def _build_alpaca(cfg: DictConfig):
    path = Path(cfg.dataset.path) / "train.jsonl"
    seq_length = cfg.dataset.max_length
    # Build character vocabulary (simple lowercase ascii + specials)
    all_chars = [chr(i) for i in range(32, 127)]
    char2idx = {ch: idx + 2 for idx, ch in enumerate(all_chars)}
    char2idx["<pad>"] = 0
    char2idx["<unk>"] = 1

    full_dataset = AlpacaCharDataset(path, seq_length, char2idx)
    val_size = int((1 - cfg.dataset.split.train) * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(cfg.training.seed))

    sample_inputs = torch.randint(0, len(char2idx), (1, seq_length))

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    return train_loader, val_loader, sample_inputs
