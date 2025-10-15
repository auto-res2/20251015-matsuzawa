import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Tuple, Any

import hydra
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from .model import build_model
from .preprocess import build_dataloaders

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class WandbLogger:
    """A thin wrapper that is a no-op when WandB is disabled."""

    def __init__(self, cfg):
        self._enabled = cfg.wandb.mode != "disabled"
        if self._enabled:
            import wandb

            self.run = wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.run_id,
                config=OmegaConf.to_container(cfg, resolve=True),
                mode=cfg.wandb.mode,
            )
            meta_path = Path(cfg.results_dir) / "wandb_metadata.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            with meta_path.open("w") as fp:
                json.dump(
                    {
                        "wandb_entity": cfg.wandb.entity,
                        "wandb_project": cfg.wandb.project,
                        "wandb_run_id": self.run.id,
                    },
                    fp,
                )
            print(f"WandB run URL: {self.run.url}")
        else:
            self.run = None

    def log(self, *args, **kwargs):
        if self._enabled:
            import wandb

            wandb.log(*args, **kwargs)

    def upload_file(self, *args, **kwargs):
        if self._enabled:
            import wandb

            wandb.save(*args, **kwargs)

    def finish(self):
        if self._enabled:
            import wandb

            wandb.finish()


# -----------------------------------------------------------------------------
# Training & Validation loops
# -----------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    scheduler=None,
) -> float:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    for batch in loader:
        inputs, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * labels.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    elapsed = time.time() - start_time
    return total_loss / total, correct / total, elapsed


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for batch in loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    elapsed = time.time() - start_time
    return total_loss / total, correct / total, elapsed / total  # per-sample time


# -----------------------------------------------------------------------------
# Objective for Optuna
# -----------------------------------------------------------------------------

def optuna_objective(trial: optuna.Trial, cfg) -> float:
    # Mutate hyper-parameters in cfg according to the search space
    for hp_name, hp_conf in cfg.optuna.search_space.items():
        if hp_conf.type == "loguniform":
            val = trial.suggest_float(hp_name, hp_conf.low, hp_conf.high, log=True)
        elif hp_conf.type == "uniform":
            val = trial.suggest_float(hp_name, hp_conf.low, hp_conf.high)
        elif hp_conf.type == "int":
            val = trial.suggest_int(hp_name, hp_conf.low, hp_conf.high)
        elif hp_conf.type == "categorical":
            val = trial.suggest_categorical(hp_name, hp_conf.choices)
        else:
            raise ValueError(f"Unknown search space type {hp_conf.type}")
        # hierarchical assignment (may be nested like training.learning_rate)
        OmegaConf.update(cfg, hp_name, val, merge=False)

    return run_single_training(cfg, trial=trial)


# -----------------------------------------------------------------------------
# Core training routine returning primary validation metric (higher is better)
# -----------------------------------------------------------------------------

def run_single_training(cfg, trial=None) -> float:
    device = torch.device(cfg.training.device)
    set_seed(cfg.training.seed)

    train_loader, val_loader, num_classes = build_dataloaders(cfg)
    model = build_model(cfg, num_classes).to(device)

    # ------------------------------------------------------------------
    # Optimizer & Scheduler
    # ------------------------------------------------------------------
    if cfg.training.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.training.learning_rate,
            momentum=getattr(cfg.training, "momentum", 0.0),
            weight_decay=cfg.training.weight_decay,
        )
    elif cfg.training.optimizer in {"adam", "adamw"}:
        opt_class = optim.Adam if cfg.training.optimizer == "adam" else optim.AdamW
        optimizer = opt_class(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer {cfg.training.optimizer}")

    # Simple scheduler support
    if cfg.training.lr_scheduler.name == "cosine_annealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.lr_scheduler.T_max)
    elif cfg.training.lr_scheduler.name == "linear":
        # Linear warm-up + decay (implemented via LambdaLR)
        def lr_lambda(step):
            warmup = cfg.training.lr_scheduler.warmup_steps
            if step < warmup:
                return float(step) / float(max(1, warmup))
            return max(
                0.0,
                float(cfg.training.epochs * len(train_loader) - step)
                / float(max(1, cfg.training.epochs * len(train_loader) - warmup)),
            )

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_acc = 0.0
    logger = WandbLogger(cfg)
    history = []
    epochs = 1 if cfg.trial_mode else cfg.training.epochs
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _ = train_one_epoch(model, train_loader, optimizer, device, scheduler)
        val_loss, val_acc, val_inf_time = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_inference_time": val_inf_time,
            }
        )
        logger.log({**history[-1], "epoch": epoch})
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # save checkpoint
            ckp_path = Path(cfg.results_dir) / "best_model.pt"
            torch.save(model.state_dict(), ckp_path)
            logger.upload_file(str(ckp_path))

        if trial is not None:
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    logger.finish()

    # Save results JSON -------------------------------------------------
    results = {
        "run_id": cfg.run_id,
        "best_val_accuracy": best_val_acc,
        "epochs_ran": epochs,
        "history": history,
        "model_parameters": count_parameters(model),
    }
    res_path = Path(cfg.results_dir) / "results.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    with res_path.open("w") as fp:
        json.dump(results, fp, indent=2)

    # Print experiment description + numerical data --------------------
    print("#" * 80)
    print("Experiment description:")
    print(OmegaConf.to_yaml(cfg))
    print("#" * 80)
    print("Experimental results:")
    print(json.dumps(results, indent=2))

    return best_val_acc


# -----------------------------------------------------------------------------
# Entry point driven by Hydra
# -----------------------------------------------------------------------------
@hydra.main(config_path="../config/experiment", version_base=None)
def main(cfg) -> None:
    cfg.results_dir = to_absolute_path(cfg.results_dir)
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    # Adapt config for trial_mode ------------------------------------------------
    if cfg.trial_mode:
        cfg.training.epochs = 1
        cfg.optuna.n_trials = 0

    # -------------------------------------------------------------------------
    # If Optuna is enabled, perform optimisation, else run once
    # -------------------------------------------------------------------------
    best_metric = None
    if cfg.optuna.n_trials > 0 and not cfg.trial_mode:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: optuna_objective(trial, cfg.copy()), n_trials=cfg.optuna.n_trials)
        best_metric = study.best_value
        # Save study
        study_path = Path(cfg.results_dir) / "optuna_study.pkl"
        optuna.study.persist_study(study, study_path)
    else:
        best_metric = run_single_training(cfg)

    # Done --------------------------------------------------------------------
    print(json.dumps({"run_id": cfg.run_id, "best_val_metric": best_metric}))


if __name__ == "__main__":
    main()
