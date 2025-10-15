import math
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig

# ----------------- MobileNetV2 Implementation -----------------


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
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
    def __init__(self, num_classes: int = 1000, width_mult: float = 1.0, dropout: float = 0.2):
        super().__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # First layer
        input_channel = int(input_channel * width_mult)
        self.features = [ConvBNReLU(3, input_channel, stride=2)]
        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # Building last several layers
        last_channel = int(last_channel * max(1.0, width_mult))
        self.features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        # Make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])  # global average pooling
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


# ----------------- Tiny DistilBERT‐like Transformer -----------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 1 x max_len x d_model
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: batch x seq x d
        x = x + self.pe[:, : x.size(1)]
        return x


class DistilBertClassifier(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=258)
        self.pos_enc = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=hidden_size // 64, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        # input_ids: batch x seq_len
        x = self.embed(input_ids)
        x = self.pos_enc(x)
        x = self.encoder(x)
        cls_token_state = x[:, 0]  # use first token
        logits = self.classifier(cls_token_state)
        return logits


# ----------------- Factory -----------------

def build_model(cfg: DictConfig, num_classes: int):
    model_name = cfg.model.name.lower()
    if "mobilenet" in model_name:
        model = MobileNetV2(num_classes=num_classes, width_mult=cfg.model.width_mult, dropout=cfg.model.dropout)
    elif "distilbert" in model_name:
        model = DistilBertClassifier(
            vocab_size=259,
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            num_classes=num_classes,
            dropout=cfg.model.dropout,
        )
    else:
        raise ValueError(f"Unknown model name {cfg.model.name}")
    return model
