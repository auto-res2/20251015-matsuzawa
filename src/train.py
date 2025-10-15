"""src/train.py
Training script. This file is never imported by other modules – it is always
executed as a separate Python process launched from main.py so that Hydra can
re-instantiate the complete configuration tree for the single run only.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import numpy as np
import optuna
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score

from . import model as models  # noqa: F401 – needed for Hydra instantiate
from . import preprocess  # noqa: F401

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def _move_batch(batch: Tuple, device: torch.device):
    """Recursively move tensors to device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return tuple(_move_batch(b, device) for b in batch)
    return batch


# -----------------------------------------------------------------------------
# Loss + metric dispatchers
# -----------------------------------------------------------------------------

def _compute_metrics(task: str, preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    if task == "classification":
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="weighted")
        return {"accuracy": float(acc), "f1_score": float(f1)}
    raise ValueError(f"Unknown task {task}")


# -----------------------------------------------------------------------------
# Core single-run training routine
# -----------------------------------------------------------------------------

def _train_single_run(cfg: DictConfig) -> Dict[str, Any]:
    _set_seed(42)

    device = torch.device(cfg.training.device)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, task_type, num_classes = preprocess.build_dataloaders(cfg)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = models.build_model(cfg, num_classes=num_classes).to(device)

    # ------------------------------------------------------------------
    # Optimiser + scheduler
    # ------------------------------------------------------------------
    if cfg.training.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.learning_rate,
            momentum=getattr(cfg.training, "momentum", 0.9),
            weight_decay=getattr(cfg.training, "weight_decay", 0.0),
        )
    elif cfg.training.optimizer in ["adam", "adamw"]:
        optim_class = torch.optim.AdamW if cfg.training.optimizer == "adamw" else torch.optim.Adam
        optimizer = optim_class(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=getattr(cfg.training, "weight_decay", 0.0),
        )
    else:
        raise ValueError(f"Unsupported optimiser {cfg.training.optimizer}")

    scheduler = None
    if "lr_scheduler" in cfg.training and cfg.training.lr_scheduler is not None:
        sched_type = cfg.training.lr_scheduler.type.lower()
        if sched_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.training.epochs
            )
        elif sched_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=cfg.training.lr_scheduler.step_size, gamma=0.1
            )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # WANDB
    # ------------------------------------------------------------------
    use_wandb = cfg.wandb.mode in ["online", "offline"]
    if use_wandb:
        import wandb

        wandb.init(
            entity="gengaru617",
            project="251015-test",
            name=cfg.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )
        Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(cfg.results_dir) / "wandb_metadata.json", "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "wandb_entity": "gengaru617",
                    "wandb_project": "251015-test",
                    "wandb_run_id": wandb.run.id,
                },
                fp,
                indent=2,
            )
        print(f"WandB URL: {wandb.run.get_url()}")
    else:
        wandb = None  # type: ignore

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_metric = 0.0
    history = []

    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        train_losses = []
        all_preds = []
        all_tgts = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            batch = _move_batch(batch, device)
            *inputs, labels = batch
            outputs = model(*inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(cfg.training, "grad_clip_norm", 5.0))
            optimizer.step()
            if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            train_losses.append(loss.item())
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.append(preds)
            all_tgts.append(labels.detach().cpu().numpy())

        train_loss = float(np.mean(train_losses))
        train_preds = np.concatenate(all_preds)
        train_tgts = np.concatenate(all_tgts)
        train_metrics = _compute_metrics(task_type, train_preds, train_tgts)

        # ------------------------- validation -------------------------
        model.eval()
        val_losses = []
        v_preds, v_tgts = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch(batch, device)
                *inputs, labels = batch
                outputs = model(*inputs)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                v_preds.append(outputs.argmax(dim=1).cpu().numpy())
                v_tgts.append(labels.cpu().numpy())
        val_loss = float(np.mean(val_losses))
        val_preds = np.concatenate(v_preds)
        val_tgts = np.concatenate(v_tgts)
        val_metrics = _compute_metrics(task_type, val_preds, val_tgts)

        current_metric = val_metrics[cfg.optuna.metric] if cfg.optuna.metric in val_metrics else val_metrics["accuracy"]
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            # checkpoint
            ckpt_path = Path(cfg.results_dir) / cfg.run_id / "best_model.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_record)

        # Logging -----------------------------------------------------------------
        if wandb is not None:
            wandb.log(epoch_record)

        print(json.dumps({"run_id": cfg.run_id, **epoch_record}))

        # scheduler plateau step
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

    # ------------------------------------------------------------------
    # save history + summary
    # ------------------------------------------------------------------
    res_dir = Path(cfg.results_dir) / cfg.run_id
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "history.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    summary = {
        "run_id": cfg.run_id,
        "best_val_metric": best_val_metric,
        "metric_name": cfg.optuna.metric,
    }
    with open(res_dir / "results.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    if wandb is not None:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(str(ckpt_path))
        wandb.log_artifact(artifact)
        wandb.finish()

    return summary


# -----------------------------------------------------------------------------
# Optuna optimisation wrapper
# -----------------------------------------------------------------------------

def _run_optuna(cfg: DictConfig):
    def objective(trial: optuna.Trial):
        sampled_cfg = deepcopy(cfg)
        for param_name, spec in cfg.optuna.search_space.items():
            if spec.type == "loguniform":
                value = trial.suggest_float(param_name, spec.low, spec.high, log=True)
            elif spec.type == "uniform":
                value = trial.suggest_float(param_name, spec.low, spec.high)
            elif spec.type == "categorical":
                value = trial.suggest_categorical(param_name, spec.choices)
            else:
                raise ValueError(f"Unsupported search space type {spec.type}")
            # We need to set the value into copied cfg – navigate dotted path.
            OmegaConf.update(sampled_cfg, f"training.{param_name}" if param_name in sampled_cfg.training else param_name, value, merge=False)

        result = _train_single_run(sampled_cfg)
        return result["best_val_metric"]

    study = optuna.create_study(direction=cfg.optuna.direction)
    study.optimize(objective, n_trials=cfg.optuna.n_trials, timeout=cfg.optuna.timeout)

    # Save study
    study_path = Path(cfg.results_dir) / cfg.run_id / "optuna_study.pkl"
    study_path.parent.mkdir(parents=True, exist_ok=True)
    optuna.study.save_study(study, study_path)

    best_cfg = deepcopy(cfg)
    for k, v in study.best_params.items():
        target = f"training.{k}" if k in cfg.training else k
        OmegaConf.update(best_cfg, target, v, merge=False)

    # retrain with best hyper-parameters
    return _train_single_run(best_cfg)


# -----------------------------------------------------------------------------
# Hydra entry-point
# -----------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def _main(cfg: DictConfig):  # noqa: D401
    """Entrypoint launched by main.py."""

    # --- trial-mode overrides --------------------------------------------------
    if cfg.trial_mode:
        OmegaConf.update(cfg, "training.epochs", 1, merge=False)
        OmegaConf.update(cfg, "optuna.n_trials", 0, merge=False)

    # --- results dir ----------------------------------------------------------
    cfg.results_dir = str(Path(hydra.utils.get_original_cwd()) / cfg.results_dir)

    print("\n========================================")
    print("Experiment description:")
    print(OmegaConf.to_yaml(cfg))
    print("========================================\n")

    # Optuna or single run
    if cfg.optuna.n_trials > 0:
        summary = _run_optuna(cfg)
    else:
        summary = _train_single_run(cfg)

    # final json summary to stdout
    print(json.dumps({"run_id": cfg.run_id, **summary}))


if __name__ == "__main__":
    _main()
