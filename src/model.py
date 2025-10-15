"""src/model.py
Implements architectures *from scratch*.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

# -----------------------------------------------------------------------------
# MobileNetV2 from scratch
# -----------------------------------------------------------------------------


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        layers.extend(
            [
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0, input_channel: int = 32):
        super().__init__()
        block = InvertedResidual
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, 8)
        layers: list[nn.Module] = [
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]
        # building inverted residual blocks
        output_channel = input_channel
        for t, c, n, s in interverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        layers.extend(
            [
                nn.Conv2d(input_channel, 1280, 1, 1, 0, bias=False),
                nn.BatchNorm2d(1280),
                nn.ReLU6(inplace=True),
            ]
        )
        self.features = nn.Sequential(*layers)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1280, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# -----------------------------------------------------------------------------
# DistilBERT-like encoder from scratch
# -----------------------------------------------------------------------------


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, None, :] == 0, float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float, ffn_ratio: int = 4):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(hidden_size, num_heads, dropout)
        self.attn_ln = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ffn_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size * ffn_ratio, hidden_size),
            nn.Dropout(dropout),
        )
        self.ffn_ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.self_attn(self.attn_ln(x), mask))
        x = x + self.ffn(self.ffn_ln(x))
        return x


class DistilEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)]
        )
        self.final_ln = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.final_ln(x)


# -----------------------------------------------------------------------------
# Text classifier with learned word + positional embeddings
# -----------------------------------------------------------------------------


class DistilBERTTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        num_classes: int = 2,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.encoder = DistilEncoder(hidden_size, num_layers, num_heads, dropout)
        self.cls_head = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.size()
        pos_ids = torch.arange(0, T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)
        x = self.dropout(x)
        x = self.encoder(x, attention_mask)
        cls = x[:, 0]  # use first token as CLS
        return self.cls_head(cls)


# -----------------------------------------------------------------------------
# Vision patch encoder based on DistilEncoder
# -----------------------------------------------------------------------------


class DistilBERTVisionClassifier(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert image_size % patch_size == 0, "image dimensions must be divisible by patch size"
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        self.patch_proj = nn.Linear(self.patch_dim, hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_size))
        self.dropout = nn.Dropout(dropout)

        self.encoder = DistilEncoder(hidden_size, num_layers, num_heads, dropout)
        self.mlp_head = nn.Linear(hidden_size, num_classes)

        self.patch_size = patch_size

        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def _patchify(self, images: torch.Tensor):
        B, C, H, W = images.size()
        p = self.patch_size
        patches = images.unfold(2, p, p).unfold(3, p, p)  # B C H/P W/P p p
        patches = patches.contiguous().view(B, C, -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4).flatten(2)  # B num_patches patch_dim
        return patches

    def forward(self, images):
        B = images.size(0)
        patches = self._patchify(images)
        x = self.patch_proj(patches)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_emb[:, : x.size(1)]
        x = self.dropout(x)
        x = self.encoder(x)
        cls = x[:, 0]
        return self.mlp_head(cls)


# -----------------------------------------------------------------------------
# Model factory
# -----------------------------------------------------------------------------

def build_model(cfg: DictConfig, num_classes: int):
    model_name = cfg.model.name.lower()
    if model_name == "mobilenetv2":
        return MobileNetV2(num_classes=num_classes, width_mult=cfg.model.width_multiplier)

    elif model_name == "distilbert":
        # Decide vision vs text by dataset.
        if cfg.dataset.name.lower() == "cifar-10":
            return DistilBERTVisionClassifier(
                image_size=cfg.dataset.image_size,
                patch_size=cfg.dataset.patch_size,
                hidden_size=cfg.model.hidden_size,
                num_layers=cfg.model.num_hidden_layers,
                num_classes=num_classes,
            )
        else:
            # Need tokenizer to get vocab size – approximate large enough.
            vocab_size = 30522  # BERT uncased vocab size
            return DistilBERTTextClassifier(
                vocab_size=vocab_size,
                hidden_size=cfg.model.hidden_size,
                num_layers=cfg.model.num_hidden_layers,
                num_classes=num_classes,
                max_len=cfg.dataset.max_seq_length,
            )

    else:
        raise ValueError(f"Unsupported model {cfg.model.name}")
