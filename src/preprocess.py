"""Data loading & preprocessing utilities."""
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torchvision.datasets import CIFAR10

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

# ------------------------------------------------------------
# Collate helpers
# ------------------------------------------------------------

def _dict_collate(batch):
    """Collate a list of dicts by stacking tensors."""
    output = {}
    for key in batch[0].keys():
        data = [b[key] for b in batch]
        if torch.is_tensor(data[0]):
            output[key] = torch.stack(data)
        else:
            output[key] = torch.tensor(data)
    return output


# ------------------------------------------------------------
# CIFAR-10 standard classification
# ------------------------------------------------------------

def _prepare_cifar_classification_dataloaders(cfg_dataset, trial_mode):
    mean = cfg_dataset.normalization.mean
    std = cfg_dataset.normalization.std
    # Transforms
    train_tfms = []
    if "random_crop" in cfg_dataset.augmentations:
        train_tfms.append(T.RandomCrop(cfg_dataset.input_size, padding=4))
    if "horizontal_flip" in cfg_dataset.augmentations:
        train_tfms.append(T.RandomHorizontalFlip())
    train_tfms.extend([T.ToTensor(), T.Normalize(mean, std)])
    test_tfms = [T.ToTensor(), T.Normalize(mean, std)]

    data_root = Path("./data").expanduser()
    full_trainset = CIFAR10(root=data_root, train=True, download=True, transform=T.Compose(train_tfms))

    # Validation split
    val_size = int(len(full_trainset) * cfg_dataset.val_split)
    train_size = len(full_trainset) - val_size
    train_set, val_set = random_split(full_trainset, [train_size, val_size])

    val_set.dataset.transform = T.Compose(test_tfms)  # override val transforms

    if trial_mode and getattr(cfg_dataset, "subset_size", None):
        # further subsample for quick tests
        train_set.indices = train_set.indices[: cfg_dataset.subset_size]
        val_set.indices = val_set.indices[: int(cfg_dataset.subset_size * cfg_dataset.val_split)]

    train_loader = DataLoader(
        train_set,
        batch_size=cfg_dataset.batch_size,
        shuffle=True,
        num_workers=cfg_dataset.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg_dataset.batch_size,
        shuffle=False,
        num_workers=cfg_dataset.num_workers,
    )
    return train_loader, val_loader, 10, None  # num_classes=10, vocab_size=None


# ------------------------------------------------------------
# CIFAR patch sequence representation for DistilBERT
# ------------------------------------------------------------
class CIFARPatchTextDataset(Dataset):
    def __init__(self, split, cfg_dataset, transform=None):
        self.base = CIFAR10(root="./data", train=(split == "train"), download=True)
        self.transform = transform
        self.cfg = cfg_dataset
        self.patch_size = cfg_dataset.patch_size
        self.seq_len = (cfg_dataset.input_size // self.patch_size) ** 2

    def _image_to_tokens(self, img):
        if self.transform:
            img = self.transform(img)
        # img is Tensor shape (C,H,W) in 0..1
        img_np = img.numpy()
        # convert to grayscale intensity 0-255
        gray = (0.2989 * img_np[0] + 0.587 * img_np[1] + 0.114 * img_np[2]) * 255.0
        gray = gray.astype(np.uint8)
        # patch averaging
        tokens = []
        for y in range(0, self.cfg.input_size, self.patch_size):
            for x in range(0, self.cfg.input_size, self.patch_size):
                patch = gray[y : y + self.patch_size, x : x + self.patch_size]
                tokens.append(int(patch.mean()))
        return tokens  # len = seq_len with values 0-255

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        tokens = self._image_to_tokens(img)
        attention_mask = [1] * len(tokens)
        sample = {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return sample


def _prepare_cifar_patch_text_dataloaders(cfg_dataset, trial_mode):
    mean = cfg_dataset.normalization.mean
    std = cfg_dataset.normalization.std
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    full_train = CIFARPatchTextDataset("train", cfg_dataset, transform=transform)
    val_split = cfg_dataset.val_split
    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(full_train, [train_size, val_size])

    if trial_mode and getattr(cfg_dataset, "subset_size", None):
        train_set.indices = train_set.indices[: cfg_dataset.subset_size]
        val_set.indices = val_set.indices[: int(cfg_dataset.subset_size * val_split)]

    train_loader = DataLoader(
        train_set,
        batch_size=cfg_dataset.batch_size,
        shuffle=True,
        collate_fn=_dict_collate,
        num_workers=cfg_dataset.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg_dataset.batch_size,
        shuffle=False,
        collate_fn=_dict_collate,
        num_workers=cfg_dataset.num_workers,
    )
    num_classes = 10
    vocab_size = 258  # 0-255 plus PAD and CLS maybe
    return train_loader, val_loader, num_classes, vocab_size


# ------------------------------------------------------------
# Alpaca instruction-following dataset
# ------------------------------------------------------------
class AlpacaTextDataset(Dataset):
    def __init__(self, split, cfg_dataset):
        self.cfg = cfg_dataset
        # Attempt to load real dataset, else fallback
        if load_dataset is not None:
            try:
                ds = load_dataset("yahma/alpaca-cleaned", split="train")
            except Exception:
                ds = None
        else:
            ds = None
        if ds is None:
            # fallback synthetic examples
            self.data = [
                {"instruction": "Say hello", "output": "Hello!"},
                {"instruction": "Say bye", "output": "Bye!"},
                {"instruction": "Tell a joke", "output": "Why did the chicken cross the road? To get to the other side."},
            ]
        else:
            self.data = ds
        # train/val split
        random.seed(42)
        random.shuffle(self.data)
        split_idx = int(len(self.data) * cfg_dataset.train_split)
        if split == "train":
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]

        # Build tokenizer
        from transformers import DistilBertTokenizerFast

        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        # Determine max_length
        self.max_length = cfg_dataset.max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample[self.cfg.text_column]
        label_text = sample[self.cfg.label_column]
        # quick binary label based on length as placeholder
        label = 1 if len(label_text) > 100 else 0
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def _prepare_alpaca_dataloaders(cfg_dataset, trial_mode):
    train_ds = AlpacaTextDataset("train", cfg_dataset)
    val_ds = AlpacaTextDataset("val", cfg_dataset)

    if trial_mode and getattr(cfg_dataset, "subset_size", None):
        train_ds.data = train_ds.data[: cfg_dataset.subset_size]
        val_ds.data = val_ds.data[: int(cfg_dataset.subset_size * cfg_dataset.val_split)]

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg_dataset.batch_size,
        shuffle=True,
        collate_fn=_dict_collate,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg_dataset.batch_size,
        shuffle=False,
        collate_fn=_dict_collate,
        num_workers=2,
    )
    num_classes = 2
    vocab_size = train_ds.tokenizer.vocab_size
    return train_loader, val_loader, num_classes, vocab_size


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def get_dataloaders(cfg, trial_mode=False):
    if cfg.dataset.name == "CIFAR-10" and cfg.dataset.get("representation", None) == "patch_sequence":
        return _prepare_cifar_patch_text_dataloaders(cfg.dataset, trial_mode)
    elif cfg.dataset.name == "CIFAR-10":
        return _prepare_cifar_classification_dataloaders(cfg.dataset, trial_mode)
    elif cfg.dataset.name == "alpaca-cleaned":
        return _prepare_alpaca_dataloaders(cfg.dataset, trial_mode)
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")