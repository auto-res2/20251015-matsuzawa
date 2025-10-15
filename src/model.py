import math
import torch
from torch import nn
import torchvision.models as tv_models

try:
    from transformers import DistilBertConfig, DistilBertModel, DistilBertForSequenceClassification
except ImportError:
    DistilBertConfig = None  # placeholder for type checker

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def compute_model_size(model):
    params = sum(p.numel() for p in model.parameters())
    size_mb = params * 4 / (1024 ** 2)  # assuming float32
    return round(size_mb, 4)


# ------------------------------------------------------------
# Model builders
# ------------------------------------------------------------

def _build_mobilenet(cfg_model, num_classes):
    # Pretrained weights only available for width_mult=1.0
    use_pretrained = cfg_model.pretrained and cfg_model.width_multiplier == 1.0
    net = tv_models.mobilenet_v2(pretrained=use_pretrained, width_mult=cfg_model.width_multiplier)
    in_feats = net.classifier[1].in_features
    net.classifier[1] = nn.Linear(in_feats, num_classes)
    return net


def _build_distilbert(cfg_model, num_classes, vocab_size):
    # custom or pretrained base
    if cfg_model.pretrained:
        config = DistilBertConfig.from_pretrained(cfg_model.name)
        config.num_labels = num_classes
        if vocab_size is not None:
            config.vocab_size = vocab_size
        model = DistilBertForSequenceClassification.from_pretrained(cfg_model.name, config=config)
    else:
        config = DistilBertConfig(
            vocab_size=vocab_size,
            hidden_size=cfg_model.hidden_size,
            n_layers=cfg_model.num_layers,
            num_labels=num_classes,
        )
        model = DistilBertForSequenceClassification(config)
    return model


# ------------------------------------------------------------
# Public builder
# ------------------------------------------------------------

def build_model(cfg, num_classes, vocab_size=None):
    name = cfg.model.name.lower()
    if "mobilenet" in name:
        return _build_mobilenet(cfg.model, num_classes)
    elif "distilbert" in name:
        return _build_distilbert(cfg.model, num_classes, vocab_size)
    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}")