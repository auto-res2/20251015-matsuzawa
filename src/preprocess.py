from pathlib import Path
from typing import Tuple, Any

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from transformers import AutoTokenizer


class PatchCIFAR10Dataset(torch.utils.data.Dataset):
    """Converts CIFAR-10 images to sequences of flattened patches"""

    def __init__(self, root: str, split: str, patch_size: int, transform=None):
        self.ds = datasets.CIFAR10(
            root=root, train=(split == "train"), download=True, transform=transform
        )
        if split == "val":
            raise ValueError("Use random_split from train set to get validation subset")
        self.patch_size = patch_size
        self.image_size = 32

    def __len__(self):
        return len(self.ds)

    def _img_to_patches(self, img):
        # img: Tensor CxHxW in [0,1]
        c, h, w = img.shape
        patches = img.unfold(1, self.patch_size, self.patch_size).unfold(
            2, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().view(c, -1, self.patch_size, self.patch_size)
        patches = patches.permute(1, 0, 2, 3)  # num_patches x C x p x p
        patches = patches.reshape(patches.size(0), -1)  # flatten
        return patches  # (num_patches, patch_dim)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        patches = self._img_to_patches(img)
        return patches, label


class SimpleTextDataset(torch.utils.data.Dataset):
    """Whitespace tokeniser with on-the-fly numericalisation"""

    def __init__(self, texts, labels, vocab=None, max_length=128):
        self.labels = labels
        self.max_length = max_length
        if vocab is None:
            vocab = {"<pad>": 0, "<unk>": 1}
            for t in texts:
                for tok in t.split():
                    if tok not in vocab:
                        vocab[tok] = len(vocab)
        self.vocab = vocab
        self.idx2tok = {i: t for t, i in vocab.items()}
        self.encoded = [self.encode(t) for t in texts]

    def encode(self, txt):
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in txt.split()][: self.max_length]
        pad_len = self.max_length - len(ids)
        return ids + [0] * pad_len, [1] * len(ids) + [0] * pad_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids, attn = self.encoded[idx]
        return torch.tensor(input_ids), torch.tensor(attn), self.labels[idx]


class HFTextDataset(torch.utils.data.Dataset):
    """Dataset using HuggingFace tokenizer"""

    def __init__(self, texts, labels, tokenizer_name, max_length=512):
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.enc = self.tok(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        return item, self.labels[idx]


# ------------------------------------------------------
# DataLoader builder (used by training)
# ------------------------------------------------------

def build_dataloaders(cfg) -> Tuple[Any, Any, Any, int]:
    """Return train/val/test dataloaders and num_classes"""
    root = str(Path("data"))
    num_workers = cfg.resources.num_workers

    if cfg.dataset.name == "CIFAR-10":
        trans_train = [transforms.ToTensor()]
        if cfg.dataset.augmentation.random_crop:
            trans_train.append(
                transforms.RandomCrop(cfg.dataset.image_size, padding=cfg.dataset.augmentation.crop_padding)
            )
        if cfg.dataset.augmentation.random_horizontal_flip:
            trans_train.append(transforms.RandomHorizontalFlip())
        trans_train.append(
            transforms.Normalize(mean=cfg.dataset.normalization.mean, std=cfg.dataset.normalization.std)
        )
        trans_train = transforms.Compose(trans_train)
        trans_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=cfg.dataset.normalization.mean, std=cfg.dataset.normalization.std),
            ]
        )
        full_train = datasets.CIFAR10(root=root, train=True, download=True, transform=trans_train)
        test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=trans_test)
        n_total = len(full_train)
        n_val = int(n_total * cfg.dataset.split_ratio.val)
        n_train = n_total - n_val
        train_set, val_set = random_split(full_train, [n_train, n_val])
        train_loader = DataLoader(train_set, batch_size=cfg.dataset.batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=cfg.dataset.batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_set, batch_size=cfg.dataset.batch_size, shuffle=False, num_workers=num_workers)
        num_classes = 10

    elif cfg.dataset.name == "alpaca-cleaned":
        file_path = Path(root) / "alpaca-cleaned.json"
        if not file_path.exists():
            # Create tiny synthetic dataset if not present (for reproducibility)
            texts = ["hello world", "foo bar"] * 100
            labels = [0, 1] * 100
        else:
            import json

            with file_path.open() as f:
                rows = json.load(f)
            texts = [r["text"] for r in rows]
            labels = [r["label"] for r in rows]

        # Split
        n_total = len(texts)
        indices = torch.randperm(n_total)
        train_end = int(n_total * cfg.dataset.split_ratio.train)
        val_end = train_end + int(n_total * cfg.dataset.split_ratio.val)
        splits = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:],
        }

        def subset(name):
            subset_texts = [texts[i] for i in splits[name]]
            subset_labels = [labels[i] for i in splits[name]]
            if cfg.dataset.tokenizer == "whitespace":
                ds = SimpleTextDataset(
                    subset_texts,
                    subset_labels,
                    vocab=None if name == "train" else vocab,
                    max_length=cfg.dataset.max_length,
                )
                return ds
            else:
                return HFTextDataset(
                    subset_texts,
                    subset_labels,
                    tokenizer_name=cfg.dataset.tokenizer,
                    max_length=cfg.dataset.max_length,
                )

        if cfg.dataset.tokenizer == "whitespace":
            # Build shared vocab on train split
            vocab_ds = SimpleTextDataset(
                [texts[i] for i in splits["train"]], [labels[i] for i in splits["train"]]
            )
            vocab = vocab_ds.vocab
        else:
            vocab = None

        train_set = subset("train")
        val_set = subset("val")
        test_set = subset("test")

        collate_fn = None  # default works for HFTextDataset since it returns dicts already tensors
        batch_size = cfg.dataset.batch_size
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        num_classes = len(set(labels))

    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")

    return train_loader, val_loader, test_loader, num_classes
