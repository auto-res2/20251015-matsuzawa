import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
import optuna
import torch
import torch.nn.functional as F
import wandb
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader

from .model import build_model
from .preprocess import build_dataloaders, seed_everything

# ----------------- Helper functions -----------------

def save_wandb_metadata(run_dir: Path, iteration: int, wandb_run: wandb.run) -> None:
    meta_dir = Path(get_original_cwd()) / f".research/iteration{iteration}"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "wandb_metadata.json"
    metadata = {
        "wandb_entity": wandb_run.entity,
        "wandb_project": wandb_run.project,
        "wandb_run_id": wandb_run.id,
    }
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    loss_meter, correct, total = 0.0, 0, 0
    for batch in loader:
        inputs, labels = batch["inputs"].to(device), batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return {
        "loss": loss_meter / total,
        "accuracy": correct / total,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    loss_meter, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch["inputs"].to(device), batch["labels"].to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss_meter += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return {
        "loss": loss_meter / total,
        "accuracy": correct / total,
    }


# ----------------- Optuna objective -----------------

def objective(trial: optuna.Trial, cfg: DictConfig) -> float:
    # Sample hyper-parameters according to cfg.optuna.search_space
    for key, space in cfg.optuna.search_space.items():
        if space["type"] == "loguniform":
            sampled = trial.suggest_float(key, space["low"], space["high"], log=True)
        elif space["type"] == "uniform":
            sampled = trial.suggest_float(key, space["low"], space["high"], log=False)
        elif space["type"] == "categorical":
            sampled = trial.suggest_categorical(key, space["choices"])
        else:
            raise ValueError(f"Unsupported optuna space type: {space['type']}")
        OmegaConf.update(cfg, key, sampled, merge=False)

    # Build dataloaders/model
    train_loader, val_loader, num_classes = build_dataloaders(cfg)
    model = build_model(cfg, num_classes)
    device = torch.device("cpu") if cfg.training.compute_device == "cpu" or not torch.cuda.is_available() else torch.device("cuda")
    model.to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.get("weight_decay", 0.0),
    )

    for _ in range(min(5, cfg.training.epochs)):  # Restrict for efficiency inside Optuna
        train_one_epoch(model, train_loader, optimizer, device)
    metrics = evaluate(model, val_loader, device)
    return metrics["loss"]


# ----------------- Main training entry -----------------

@hydra.main(config_path="../config", config_name="config", version_base=None)
def train_main(cfg: DictConfig) -> None:
    original_cwd = Path(get_original_cwd())

    # Handle trial_mode overrides
    if cfg.get("trial_mode", False):
        cfg.training.epochs = 1
        cfg.optuna.n_trials = 0
        cfg.training.batch_size = min(8, cfg.training.batch_size)

    seed_everything(42)

    # Build dataloaders
    train_loader, val_loader, num_classes = build_dataloaders(cfg)

    # Build model
    model = build_model(cfg, num_classes)

    device = torch.device("cpu") if cfg.training.compute_device == "cpu" or not torch.cuda.is_available() else torch.device("cuda")
    model.to(device)

    # Optimizer
    if cfg.training.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.training.learning_rate,
            momentum=cfg.training.get("momentum", 0.0),
            weight_decay=cfg.training.get("weight_decay", 0.0),
        )
    elif cfg.training.optimizer in ["adam", "adamw"]:
        optimizer_cls = optim.AdamW if cfg.training.optimizer == "adamw" else optim.Adam
        optimizer = optimizer_cls(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.get("weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Unsupported optimizer {cfg.training.optimizer}")

    # LR Scheduler (simple cosine)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs) if cfg.training.lr_scheduler == "cosine" else None

    # Initialise WandB
    wandb_run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.run_id,
        tags=[cfg.method, cfg.dataset.name, cfg.model.name],
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )
    print(f"WandB URL: {wandb_run.url}")

    # Save WandB metadata
    iteration = int(os.environ.get("EXPERIMENT_ITERATION", "0"))
    run_dir = Path(cfg.results_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_wandb_metadata(run_dir, iteration, wandb_run)

    # Optuna hyper-parameter optimisation
    if cfg.optuna.n_trials > 0 and not cfg.get("trial_mode", False):
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, cfg), n_trials=cfg.optuna.n_trials)
        best_params = study.best_params
        print(f"Best Optuna params: {best_params}")
        for k, v in best_params.items():
            OmegaConf.update(cfg, k, v, merge=False)
        # rebuild with best parameters
        train_loader, val_loader, num_classes = build_dataloaders(cfg)
        model = build_model(cfg, num_classes)
        model.to(device)
        optimizer.param_groups[0]["lr"] = cfg.training.learning_rate

    # Training loop
    history = []
    for epoch in range(1, cfg.training.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        if scheduler is not None:
            scheduler.step()
        epoch_metrics = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_metrics)
        # Log to WandB
        wandb_run.log(epoch_metrics)
        print(json.dumps(epoch_metrics))

    # Save final checkpoint & metrics
    chkpt_path = run_dir / "model_final.pt"
    torch.save(model.state_dict(), chkpt_path)
    wandb_run.save(str(chkpt_path))

    final_results = {
        "run_id": cfg.run_id,
        "final_epoch": history[-1]["epoch"],
        "best_val_accuracy": max(h["val_accuracy"] for h in history),
        "history": history,
    }

    with open(run_dir / "results.json", "w", encoding="utf-8") as fp:
        json.dump(final_results, fp, indent=2)

    # Print experiment description and final numeric data
    print("Experiment Description:")
    print(OmegaConf.to_yaml(cfg))
    print("Final Results:")
    print(json.dumps(final_results, indent=2))

    wandb_run.finish()


if __name__ == "__main__":
    train_main()
