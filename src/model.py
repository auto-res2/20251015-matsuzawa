"""Model implementations for MobileNetV2, DistilBERT variants and CharCNN."""
from typing import Any
import math

import torch
import torch.nn as nn
from omegaconf import DictConfig

# ----------------------------------------------------------------------------
# MobileNetV2 Implementation (2-D)
# ----------------------------------------------------------------------------


def _conv_bn(inp: int, oup: int, stride: int):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def _conv_1x1_bn(inp: int, oup: int):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(_conv_1x1_bn(inp, hidden_dim))
        layers.extend(
            [
                # depthwise 3x3
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # project
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):  # noqa: D401
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0, dropout: float = 0.2):
        super().__init__()
        # Setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        input_channel = int(32 * width_mult)
        layers: list[nn.Module] = [_conv_bn(3, input_channel, 2)]
        block = InvertedResidual
        # building inverted residual blocks
        for t, c, n, s in self.cfgs:
            output_channel = int(c * width_mult)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        layers.append(_conv_1x1_bn(input_channel, last_channel))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(last_channel, num_classes))
        self._initialize_weights()

    def forward(self, x):  # noqa: D401
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ----------------------------------------------------------------------------
# Char 1-D MobileNet-style CNN
# ----------------------------------------------------------------------------

class CharMobileNet(nn.Module):
    """1-D CNN inspired by MobileNet blocks for character sequences."""

    def __init__(self, vocab_size: int, embedding_dim: int, seq_length: int, num_classes: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Sequential(
            nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1, groups=128),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):  # noqa: D401
        x = self.embedding(x).transpose(1, 2)  # (B, E, L)
        x = self.conv(x)
        x = self.classifier(x)
        return x


# ----------------------------------------------------------------------------
# Image Transformer (DistilBERT-like) for Vision
# ----------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # noqa: D401
        x = self.proj(x)  # (B, C, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


class ImageTransformer(nn.Module):
    """Simplified Transformer encoder for image patches."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        img_size = cfg.dataset.input_size
        patch_size = cfg.model.image_patch_size
        num_patches = (img_size // patch_size) ** 2
        embed_dim = cfg.model.hidden_size
        self.patch_embed = PatchEmbed(3, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dropout=cfg.model.dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.model.num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, cfg.model.num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):  # noqa: D401
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def model_num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_model(model_cfg: DictConfig) -> nn.Module:  # noqa: C901
    name = model_cfg.name.lower()
    if name == "mobilenetv2":
        return MobileNetV2(
            num_classes=model_cfg.num_classes,
            width_mult=getattr(model_cfg, "width_multiplier", 1.0),
            dropout=model_cfg.dropout,
        )
    if name == "distilbert":
        # Distinguish between image and text variant using attribute presence
        if hasattr(model_cfg, "image_patch_size"):
            return ImageTransformer(model_cfg._get_root())
        else:
            # Text DistilBERT fine-tuning using transformers
            from transformers import DistilBertForSequenceClassification, DistilBertConfig  # type: ignore

            pretrained = getattr(model_cfg, "pretrained", None)
            if pretrained and pretrained != "false":
                model = DistilBertForSequenceClassification.from_pretrained(pretrained, num_labels=model_cfg.num_labels)
            else:
                config = DistilBertConfig(
                    vocab_size=30522,  # default BERT vocab
                    n_layers=model_cfg.num_layers,
                    dim=model_cfg.hidden_size,
                    n_heads=12,
                    num_labels=model_cfg.num_labels,
                    dropout=model_cfg.dropout,
                )
                model = DistilBertForSequenceClassification(config)
            return model
    if name == "mobilenetv2" and getattr(model_cfg, "architecture", "") == "charcnn":
        raise ValueError("architecture field should change name use CharMobileNet")

    if name == "charmobilenet":
        return CharMobileNet(
            vocab_size=model_cfg.vocab_size,
            embedding_dim=model_cfg.embedding_dim,
            seq_length=model_cfg.seq_length,
            num_classes=model_cfg.num_classes,
            dropout=model_cfg.dropout,
        )

    raise ValueError(f"Unsupported model name {model_cfg.name}")
