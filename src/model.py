import math
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertConfig, DistilBertModel


class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes: int, width_mult: float = 1.0, pretrained: bool = False):
        super().__init__()
        self.base = models.mobilenet_v2(width_mult=width_mult, pretrained=pretrained)
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base(x)


class DistilBERTTextClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        else:
            cfg = DistilBertConfig()
            self.bert = DistilBertModel(cfg)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, batch):
        if isinstance(batch, dict):
            outputs = self.bert(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            cls = outputs.last_hidden_state[:, 0]
        else:
            # whitespace dataset returns input_ids, attn, label
            input_ids, attn = batch
            outputs = self.bert(input_ids=input_ids, attention_mask=attn)
            cls = outputs.last_hidden_state[:, 0]
        return self.classifier(cls)


class DistilBERTPatchClassifier(nn.Module):
    """Vision via patch tokens fed into DistilBERT"""

    def __init__(self, num_classes: int, patch_size: int = 4, embed_dim: int = 768, pretrained: bool = False):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = 3 * patch_size * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, embed_dim)
        if pretrained:
            self.encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        else:
            cfg = DistilBertConfig(vocab_size=1, max_position_embeddings=1024, n_heads=12, dim=embed_dim)
            self.encoder = DistilBertModel(cfg)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer("position_ids", torch.arange(0, 2048).unsqueeze(0))
        self.pos_embed = nn.Embedding(2048, embed_dim)

    def forward(self, patches):
        # patches: (B, N, patch_dim)
        b, n, d = patches.size()
        x = self.patch_to_embedding(patches)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)  # B, N+1, D
        positions = self.position_ids[:, : x.size(1)]
        x = x + self.pos_embed(positions)
        attn_mask = torch.ones(b, x.size(1), dtype=torch.long, device=x.device)
        enc_out = self.encoder(inputs_embeds=x, attention_mask=attn_mask)
        return self.classifier(enc_out.last_hidden_state[:, 0])


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------


class ModelFactory:
    @staticmethod
    def build(cfg, num_classes: int):
        name = cfg.model.name.lower()
        if name == "mobilenetv2":
            return MobileNetV2Classifier(
                num_classes=num_classes,
                width_mult=cfg.model.width_multiplier,
                pretrained=cfg.model.pretrained,
            )
        elif name.startswith("distilbert") and cfg.dataset.task in (
            "text_classification",
            "instruction_classification",
        ):
            return DistilBERTTextClassifier(
                num_classes=num_classes,
                pretrained=cfg.model.pretrained,
            )
        elif name.startswith("distilbert"):
            return DistilBERTPatchClassifier(
                num_classes=num_classes,
                patch_size=cfg.model.patch_embedding.patch_size,
                embed_dim=cfg.model.patch_embedding.embedding_dim,
                pretrained=cfg.model.pretrained,
            )
        else:
            raise ValueError(f"Unsupported model: {cfg.model.name}")
