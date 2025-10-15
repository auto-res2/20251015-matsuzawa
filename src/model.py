import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Helper blocks for MobileNetV2
# -----------------------------------------------------------------------------

def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.identity = stride == 1 and inp == oup
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
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

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
        input_channel = _make_divisible(input_channel * width_mult, 4)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), 4)
        features = [
            nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, 4)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False))
        features.append(nn.BatchNorm2d(self.last_channel))
        features.append(nn.ReLU6(inplace=True))
        self.features = nn.Sequential(*features)

        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.last_channel, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        x = self.classifier(x)
        return x


# -----------------------------------------------------------------------------
# Very small Transformer encoder used for Distil-style models (image & text)
# -----------------------------------------------------------------------------

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        seq_length: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        sap: bool = False,  # Spatial average pooling flag for images
    ):
        super().__init__()
        self.sap = sap
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, seq_len] (text) or [B, seq_len, d]
        if x.dim() == 3 and self.sap:
            b, t, d = x.shape
            x = x
        else:
            x = self.token_emb(x) + self.pos_emb[:, : x.size(1)]
        x = self.encoder(x.transpose(0, 1)).mean(dim=0)
        return self.classifier(x)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # B, C, H/P, W/P
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim
        return x


class ImageTransformerClassifier(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        num_classes: int,
        embed_dim: int = 128,
        depth: int = 4,
        nhead: int = 4,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead, dim_feedforward=embed_dim * 4)
        self.encoder = nn.TransformerEncoder(encoder_layer, depth)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_emb
        x = self.encoder(x.transpose(0, 1)).mean(dim=0)
        return self.classifier(x)


# -----------------------------------------------------------------------------
# Text CNN for "mobilenet_v2_text" baseline
# -----------------------------------------------------------------------------


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int, kernel_sizes=(3, 4, 5), num_channels=100):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_channels, (k, embed_dim)) for k in kernel_sizes]
        )
        self.fc = nn.Linear(num_channels * len(kernel_sizes), num_classes)

    def forward(self, x):
        # x: [B, L]
        x = self.embed(x)  # [B, L, D]
        x = x.unsqueeze(1)  # [B, 1, L, D]
        conv_outs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        pools = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_outs]
        out = torch.cat(pools, 1)
        return self.fc(out)


# -----------------------------------------------------------------------------
# Model builder entry
# -----------------------------------------------------------------------------

def build_model(cfg, num_classes: int):
    if cfg.task == "image_classification":
        if cfg.model.name.startswith("mobilenet_v2"):
            return MobileNetV2(num_classes=num_classes, width_mult=cfg.model.width_multiplier)
        elif cfg.model.name.startswith("distilbert") or cfg.model.name.endswith("patch"):
            patch_size = getattr(cfg.model.patch_tokenizer, "patch_size", 4)
            return ImageTransformerClassifier(
                img_size=cfg.dataset.image_size,
                patch_size=patch_size,
                num_classes=num_classes,
            )
    elif cfg.task == "text_classification":
        vocab_size = getattr(cfg.model.tokenizer, "vocab_size", 30000)
        if cfg.model.name.startswith("mobilenet_v2_text"):
            return TextCNN(vocab_size=vocab_size, embed_dim=cfg.model.embedding_dim, num_classes=num_classes)
        else:
            return TransformerClassifier(
                vocab_size=vocab_size,
                num_classes=num_classes,
                seq_length=cfg.dataset.max_length,
                dropout=getattr(cfg.model, "dropout", 0.1),
            )

    raise ValueError(f"Unsupported model/task combination: {cfg.model.name} / {cfg.task}")
