import json
import math
import os
import random
import urllib.request
from pathlib import Path
from typing import Tuple

import torch
import torchvision.transforms as T
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10

# Special tokens for our simple tokenizer
CLS_ID = 256
SEP_ID = 257
PAD_ID = 258
VOCAB_SIZE = 259  # 0–255 ascii + CLS/SEP/PAD


# ----------------- Utility -----------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------- CIFAR-10 token dataset for DistilBERT -----------------

class CIFAR10TokenDataset(Dataset):
    def __init__(self, root: str, train: bool):
        self.base = CIFAR10(root=root, train=train, download=True)

    def __len__(self):
        return len(self.base)

    def _image_to_tokens(self, img) -> torch.Tensor:
        # Convert to grayscale 32x32
        img = T.Grayscale()(img)
        img = T.ToTensor()(img)  # 1 x 32 x 32, values 0–1
        img = (img * 255).long().squeeze(0)  # 32 x 32 ints
        # Mean over 4x4 patches to create 8x8
        tokens = []
        patch_size = 4
        for i in range(0, 32, patch_size):
            for j in range(0, 32, patch_size):
                patch = img[i : i + patch_size, j : j + patch_size]
                mean_val = int(torch.mean(patch).item())  # 0–255
                token_id = mean_val // 4  # Quantise to 64 bins but keep within 0–255
                tokens.append(token_id)
        tokens = [CLS_ID] + tokens + [SEP_ID]
        attn_mask = [1] * len(tokens)
        # pad to 66 tokens (already 66) but keep flexible
        return torch.tensor(tokens), torch.tensor(attn_mask)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        input_ids, attention_mask = self._image_to_tokens(img)
        return {
            "inputs": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ----------------- Alpaca cleaned dataset -----------------

class AlpacaRecord:
    def __init__(self, instruction: str, output: str):
        self.text = instruction + " " + output
        # simple binary label: long vs short output
        self.label = 1 if len(output) > 100 else 0


class AlpacaTextTokenDataset(Dataset):
    def __init__(self, split: str, max_length: int):
        # Download cleaned Alpaca dataset json if needed
        data_path = Path("alpaca_data_cleaned.json")
        if not data_path.exists():
            url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data_cleaned.json"
            urllib.request.urlretrieve(url, data_path)
        with open(data_path, "r", encoding="utf-8") as fp:
            records_json = json.load(fp)
        random.shuffle(records_json)
        n_total = len(records_json)
        val_split = int(0.05 * n_total)
        if split == "train":
            records_json = records_json[val_split:]
        else:
            records_json = records_json[:val_split]
        self.records = [AlpacaRecord(r["instruction"], r["output"]) for r in records_json]
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def _tokenize(self, text: str):
        ids = [ord(c) % 256 for c in text][: self.max_length - 2]
        ids = [CLS_ID] + ids + [SEP_ID]
        attn_mask = [1] * len(ids)
        while len(ids) < self.max_length:
            ids.append(PAD_ID)
            attn_mask.append(0)
        return torch.tensor(ids), torch.tensor(attn_mask)

    def __getitem__(self, idx):
        rec = self.records[idx]
        input_ids, attn = self._tokenize(rec.text)
        return {
            "inputs": input_ids,
            "attention_mask": attn,
            "labels": torch.tensor(rec.label, dtype=torch.long),
        }


class AlpacaTextImageDataset(Dataset):
    """Represent text as 32x32 grayscale image for MobileNet"""

    def __init__(self, split: str):
        data_path = Path("alpaca_data_cleaned.json")
        if not data_path.exists():
            url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data_cleaned.json"
            urllib.request.urlretrieve(url, data_path)
        with open(data_path, "r", encoding="utf-8") as fp:
            records_json = json.load(fp)
        random.shuffle(records_json)
        n_total = len(records_json)
        val_split = int(0.05 * n_total)
        if split == "train":
            records_json = records_json[val_split:]
        else:
            records_json = records_json[:val_split]
        self.records = records_json
        self.image_transform = T.Compose(
            [T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
        )

    def __len__(self):
        return len(self.records)

    def _text_to_image(self, text: str):
        ascii_vals = [ord(c) % 256 for c in text][: 32 * 32]
        if len(ascii_vals) < 32 * 32:
            ascii_vals += [0] * (32 * 32 - len(ascii_vals))
        img = torch.tensor(ascii_vals, dtype=torch.uint8).view(1, 32, 32)  # 1x32x32
        img = img.repeat(3, 1, 1).float() / 255.0
        return img

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_tensor = self._text_to_image(rec["instruction"] + " " + rec["output"])
        label = 1 if len(rec["output"]) > 100 else 0
        return {"inputs": img_tensor, "labels": torch.tensor(label, dtype=torch.long)}


# ----------------- DataLoader builder -----------------

def collate_fn_token(batch):
    input_ids = torch.stack([b["inputs"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"inputs": input_ids, "labels": labels}


def build_dataloaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader, int]:
    name = cfg.dataset.name
    bs = cfg.training.batch_size
    if name == "CIFAR-10":
        if "distilbert" in cfg.model.name.lower():
            train_set = CIFAR10TokenDataset(root="data", train=True)
            val_set = CIFAR10TokenDataset(root="data", train=False)
            num_classes = 10
            train_loader = DataLoader(
                train_set, batch_size=bs, shuffle=True, collate_fn=collate_fn_token
            )
            val_loader = DataLoader(
                val_set, batch_size=bs, shuffle=False, collate_fn=collate_fn_token
            )
        else:  # MobileNet and other image models
            transform_train = [T.ToTensor()]
            if cfg.dataset.augmentations.random_crop:
                transform_train.insert(0, T.RandomCrop(cfg.dataset.image_size, padding=4))
            if cfg.dataset.augmentations.random_flip:
                transform_train.append(T.RandomHorizontalFlip())
            transform_train.append(
                T.Normalize(mean=cfg.dataset.normalization.mean, std=cfg.dataset.normalization.std)
            )
            transform_train = T.Compose(transform_train)
            transform_test = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(mean=cfg.dataset.normalization.mean, std=cfg.dataset.normalization.std),
                ]
            )
            full_train = CIFAR10(root="data", train=True, download=True, transform=transform_train)
            val_size = int(cfg.dataset.val_split * len(full_train))
            train_size = len(full_train) - val_size
            train_set, val_set = random_split(full_train, [train_size, val_size])
            val_set.dataset.transform = transform_test  # type: ignore
            num_classes = 10
            train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=2)
    elif name == "alpaca-cleaned":
        if "distilbert" in cfg.model.name.lower():
            train_set = AlpacaTextTokenDataset("train", cfg.dataset.max_length)
            val_set = AlpacaTextTokenDataset("val", cfg.dataset.max_length)
            num_classes = 2
            train_loader = DataLoader(
                train_set, batch_size=bs, shuffle=True, collate_fn=collate_fn_token
            )
            val_loader = DataLoader(
                val_set, batch_size=bs, shuffle=False, collate_fn=collate_fn_token
            )
        else:  # MobileNet images from text
            train_set = AlpacaTextImageDataset("train")
            val_set = AlpacaTextImageDataset("val")
            num_classes = 2
            train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=2)
    else:
        raise ValueError(f"Unsupported dataset {name}")
    return train_loader, val_loader, num_classes
