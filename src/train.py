import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Any

import hydra
import numpy as np
import optuna
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score

from .model import ModelFactory
from .preprocess import build_dataloaders

# ===============  Utility helpers ==================

def set_seed(seed: int) -> None:
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _save_json(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


# ===============  Core training logic ==================

def _single_train(cfg, trial_params: Dict[str, Any] = None) -> Tuple[float, Dict[str, Any]]:
    """Executes one training run, possibly inside an Optuna trial"""
    if trial_params is not None:
        # Override hyper-parameters for this trial
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"training": trial_params}))

    device = torch.device(cfg.resources.device)
    train_loader, val_loader, test_loader, num_classes = build_dataloaders(cfg)

    model = ModelFactory.build(cfg, num_classes=num_classes)
    model.to(device)

    # optimizer
    if cfg.training.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.learning_rate,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
        )
    elif cfg.training.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
    else:  # adamw
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

    if cfg.training.scheduler.type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs
        )
    else:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

    # -----  Train loop -----
    best_val_acc = 0.0
    best_state = None
    epoch_metrics = []
    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            optimizer.step()
            epoch_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total
        train_loss = epoch_loss / total

        # Validation
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = F.cross_entropy(logits, labels)
                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        val_acc = val_correct / val_total
        val_f1 = f1_score(
            torch.cat(all_labels), torch.cat(all_preds), average="macro", zero_division=0
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

        scheduler.step()

        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss / val_total,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
            }
        )

    # -----  Evaluation on test -----
    model.load_state_dict(best_state)
    model.eval()
    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
    start_time = time.time()
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    inference_time = (time.time() - start_time) / test_total
    test_acc = test_correct / test_total
    test_f1 = f1_score(
        torch.cat(all_labels), torch.cat(all_preds), average="macro", zero_division=0
    )

    model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / 1024 ** 2

    metrics = {
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "inference_time": inference_time,
        "model_size_mb": model_size_mb,
        "epochs": cfg.training.epochs,
    }
    return best_val_acc, {"epoch_metrics": epoch_metrics, "final_metrics": metrics}


# ===============  Optuna objective ==================

def _objective(trial: optuna.Trial, cfg):
    # Sample hyper-parameters from search space defined in cfg.optuna.search_space
    trial_params = {}
    for hp_name, hp_def in cfg.optuna.search_space.items():
        if hp_def.type == "loguniform":
            trial_params[hp_name] = trial.suggest_float(hp_name, hp_def.low, hp_def.high, log=True)
        elif hp_def.type == "uniform":
            trial_params[hp_name] = trial.suggest_float(hp_name, hp_def.low, hp_def.high)
        elif hp_def.type == "categorical":
            trial_params[hp_name] = trial.suggest_categorical(hp_name, hp_def.choices)
        else:
            raise ValueError(f"Unsupported hp type: {hp_def.type}")
    val_acc, _ = _single_train(cfg, trial_params)
    return val_acc


# ===============  Main entrypoint ==================

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def train_app(cfg) -> None:
    """Hydra entry point called by CLI: python -m src.train run=<run_id>"""
    # -------------------------------------------------------------
    # Merge base cfg (from config/config.yaml) with run-specific cfg
    # -------------------------------------------------------------
    run_cfg_path = Path(__file__).parent.parent / "config" / "run" / f"{cfg.run}.yaml"
    if not run_cfg_path.exists():
        print(f"Run config {run_cfg_path} not found.", file=sys.stderr)
        sys.exit(1)
    run_cfg = OmegaConf.load(run_cfg_path)
    cfg = OmegaConf.merge(cfg, run_cfg)

    # Override for trial mode --------------------------------------------------
    if cfg.trial_mode:
        cfg.training.epochs = 1
        cfg.optuna.n_trials = 0

    set_seed(cfg.seed)

    # =========== Experiment description print ===============================
    print("=" * 80)
    print("Experiment description:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # ==================   WandB initialisation ===============================
    wb_run = None
    if cfg.wandb.mode != "disabled":
        import wandb

        wb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=cfg.run,
            mode=cfg.wandb.mode,
        )
        # Save metadata -------------------------------------------------------
        meta = {
            "wandb_entity": cfg.wandb.entity,
            "wandb_project": cfg.wandb.project,
            "wandb_run_id": wb_run.id,
        }
        metadata_path = Path(cfg.results_dir) / cfg.run / "wandb_metadata.json"
        _save_json(metadata_path, meta)
        print(f"Weights & Biases URL: {wb_run.url}")

    # ==================   Training / Optuna ===================================
    if cfg.optuna.n_trials > 0:
        study = optuna.create_study(direction=cfg.optuna.direction)
        study.optimize(lambda trial: _objective(trial, cfg), n_trials=cfg.optuna.n_trials)
        best_params = study.best_params
        print(f"Best hyper-parameters: {best_params}")
        _, results = _single_train(cfg, best_params)
    else:
        _, results = _single_train(cfg)

    # ==================  Save + Log ==========================================
    results_path = Path(cfg.results_dir) / cfg.run / "results.json"
    _save_json(results_path, results)

    if wb_run is not None:
        for ep in results["epoch_metrics"]:
            wandb.log({k: v for k, v in ep.items() if k != "epoch"}, step=ep["epoch"])
        wandb.log(results["final_metrics"])
        wandb.save(str(results_path))
        wb_run.finish()

    # ==================  Print metrics to STDOUT (mandatory) ==================
    print(json.dumps({"run_id": cfg.run, **results["final_metrics"]}))


if __name__ == "__main__":
    train_app()
