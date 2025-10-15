"""src/preprocess.py
Dataset & dataloader preparation utilities.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import torch
import torchvision.transforms as T
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10

# -----------------------------------------------------------------------------
# CIFAR-10 helpers
# -----------------------------------------------------------------------------


def _prepare_cifar(cfg: DictConfig) -> Tuple[DataLoader, DataLoader, int]:
    mean = cfg.dataset.normalization.mean
    std = cfg.dataset.normalization.std

    transform = T.Compose(
        [
            T.Resize(cfg.dataset.image_size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    trainset = CIFAR10(root=cfg.dataset.root, train=True, download=True, transform=transform)

    val_ratio = cfg.dataset.val_split
    val_size = int(len(trainset) * val_ratio)
    train_size = len(trainset) - val_size
    train_subset, val_subset = random_split(trainset, [train_size, val_size])

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    return train_loader, val_loader, 10


# -----------------------------------------------------------------------------
# Alpaca-cleaned – supervised fine-tuning dataset (binary classification demo)
# -----------------------------------------------------------------------------


class AlpacaClassificationDataset(Dataset):
    """A very small supervised dataset built from alpaca-cleaned JSON lines.

    Each line is expected to contain::
        {"text": "...", "label": 0/1}
    """

    def __init__(self, path: Path, tokenizer, max_length: int):
        self.samples = []
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                item = json.loads(line)
                self.samples.append((item["text"], int(item["label"])))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), torch.tensor(label, dtype=torch.long)


# -----------------------------------------------------------------------------
# DataLoader builder dispatcher
# -----------------------------------------------------------------------------

def build_dataloaders(cfg: DictConfig):
    if cfg.dataset.name.lower() == "cifar-10":
        train_loader, val_loader, num_classes = _prepare_cifar(cfg)
        task_type = "classification"
        return train_loader, val_loader, task_type, num_classes

    elif cfg.dataset.name.lower() == "alpaca-cleaned":
        from transformers import AutoTokenizer

        data_path = Path(cfg.dataset.get("path", "./data/alpaca-cleaned.json"))
        tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.tokenizer)
        full_dataset = AlpacaClassificationDataset(
            path=data_path, tokenizer=tokenizer, max_length=cfg.dataset.max_seq_length
        )
        val_size = int(len(full_dataset) * cfg.dataset.val_split)
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.dataset.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.dataset.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
        )
        return train_loader, val_loader, "classification", 2

    else:
        raise ValueError(f"Unsupported dataset {cfg.dataset.name}")
