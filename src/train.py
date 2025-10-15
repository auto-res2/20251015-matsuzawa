import os
import json
import time
import copy
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from hydra.utils import to_absolute_path
from sklearn.metrics import accuracy_score, f1_score
import wandb
import optuna

from .preprocess import get_dataloaders
from .model import build_model, compute_model_size

# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------

def _save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    preds, gts = [], []
    for batch in loader:
        optimizer.zero_grad()
        if isinstance(batch, dict):  # text-like dict batch
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = labels.to(device)
            outputs = model(**batch, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
        else:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * labels.size(0)
        preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
        gts.extend(labels.detach().cpu().tolist())
    epoch_loss /= len(loader.dataset)
    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds, average="macro")
    return epoch_loss, acc, f1


def _evaluate(model, loader, criterion, device):
    model.eval()
    preds, gts = [], []
    eval_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                labels = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = labels.to(device)
                outputs = model(**batch, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
            else:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
            eval_loss += loss.item() * labels.size(0)
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            gts.extend(labels.cpu().tolist())
    eval_loss /= len(loader.dataset)
    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds, average="macro")
    return eval_loss, acc, f1


# ------------------------------------------------------------
# Optuna objective
# ------------------------------------------------------------

def _objective_factory(base_cfg, results_dir, device):
    def objective(trial):
        cfg = copy.deepcopy(base_cfg)
        # override with suggested hyperparameters
        for key, spec in cfg.optuna.search_space.items():
            if spec["type"] == "loguniform":
                val = trial.suggest_float(key, spec["low"], spec["high"], log=True)
            elif spec["type"] == "uniform":
                val = trial.suggest_float(key, spec["low"], spec["high"])
            elif spec["type"] == "categorical":
                val = trial.suggest_categorical(key, spec["choices"])
            elif spec["type"] == "int":
                val = trial.suggest_int(key, spec["low"], spec["high"])
            else:
                raise ValueError(f"Unsupported Optuna param type: {spec['type']}")
            # write back into cfg
            if key in cfg.training:
                cfg.training[key] = val
            elif key.startswith("dataset_") and hasattr(cfg.dataset, key[8:]):
                setattr(cfg.dataset, key[8:], val)
            else:
                OmegaConf.set_struct(cfg, False)
                OmegaConf.update(cfg, key, val)
        # run a single training pass with new hyperparams
        metrics = _run_training(cfg, results_dir, device, log_to_wandb=False)
        return metrics["val_accuracy"]

    return objective


# ------------------------------------------------------------
# Core training routine (single hyper-parameter configuration)
# ------------------------------------------------------------

def _run_training(cfg, results_dir, device, log_to_wandb=True):
    # Data
    train_loader, val_loader, num_classes, vocab_size = get_dataloaders(cfg, trial_mode=cfg.trial_mode)
    # Model
    model = build_model(cfg, num_classes=num_classes, vocab_size=vocab_size)
    model.to(device)

    # Criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    if cfg.training.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.learning_rate,
            momentum=cfg.training.get("momentum", 0.0),
            weight_decay=cfg.training.weight_decay,
        )
    else:  # adamw
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

    # Scheduler
    if cfg.training.get("scheduler") == "cosine_annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
    elif cfg.training.get("scheduler") == "linear":
        total_steps = len(train_loader) * cfg.training.epochs
        warmup_steps = cfg.training.get("warmup_steps", 0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(1.0, step / max(1, warmup_steps)) if step < warmup_steps else max(
                0.0, (total_steps - step) / max(1, total_steps - warmup_steps)
            ),
        )
    else:
        scheduler = None

    # WandB
    wandb_run = None
    if log_to_wandb and cfg.wandb.mode != "disabled":
        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.wandb.get("run_name", cfg.run_id),
            tags=list(cfg.wandb.get("tags", [])),
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
            reinit=True,
        )
        # Save metadata immediately
        md_path = Path(results_dir) / "wandb_metadata.json"
        _save_json(
            {
                "wandb_entity": cfg.wandb.entity,
                "wandb_project": cfg.wandb.project,
                "wandb_run_id": wandb_run.id,
            },
            md_path,
        )
        print(f"WandB URL: {wandb_run.url}")

    # Training loop
    history = []
    best_val_acc = 0.0
    for epoch in range(cfg.training.epochs):
        start_time = time.time()
        train_loss, train_acc, train_f1 = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = _evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - start_time

        if scheduler is not None:
            scheduler.step()

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "train_f1": train_f1,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_f1": val_f1,
                    "epoch_time_sec": epoch_time,
                }
            )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "epoch_time_sec": epoch_time,
            }
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pt"))
            if wandb_run is not None:
                wandb_run.save(os.path.join(results_dir, "best_model.pt"))

    # Inference time (simple forward pass on one batch)
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader))
        start = time.time()
        if isinstance(batch, dict):
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)
        else:
            inputs, _ = batch
            _ = model(inputs.to(device))
        inference_time = time.time() - start

    model_size_mb = compute_model_size(model)

    final_results = {
        "run_id": cfg.run_id,
        "method": cfg.method,
        "model_name": cfg.model.name,
        "dataset": cfg.dataset.name,
        "val_accuracy": best_val_acc,
        "val_f1": val_f1,
        "inference_time_sec": inference_time,
        "model_size_mb": model_size_mb,
        "history": history,
    }

    if wandb_run is not None:
        wandb_run.log({"final/val_accuracy": best_val_acc, "final/model_size_mb": model_size_mb})
        wandb_run.finish()

    return final_results


# ------------------------------------------------------------
# Entry-point with Hydra
# ------------------------------------------------------------
@hydra.main(config_path="../config", config_name="config", version_base=None)
def train_app(cfg):
    """Hydra entry-point for a single training run."""
    # The run config is loaded under cfg.run by Hydra
    # Merge it with the top-level config
    OmegaConf.set_struct(cfg.run, False)
    run_cfg = OmegaConf.merge(cfg.run, OmegaConf.create({
        "results_dir": cfg.results_dir,
        "trial_mode": cfg.trial_mode,
        "wandb": cfg.wandb
    }))

    # Apply trial_mode overrides
    if cfg.trial_mode:
        run_cfg.training.epochs = 1
        run_cfg.optuna.n_trials = 0
        if run_cfg.dataset.name == "CIFAR-10":
            run_cfg.dataset.subset_size = 500  # use small subset
        elif run_cfg.dataset.name == "alpaca-cleaned":
            run_cfg.dataset.subset_size = 500

    # Make results dir
    results_dir = Path(to_absolute_path(cfg.results_dir)) / run_cfg.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Persist the merged config for record keeping
    OmegaConf.save(run_cfg, results_dir / "full_config.yaml")

    # Description (printed before results as requested)
    description = (
        f"Experiment {run_cfg.run_id}: Method={run_cfg.method}, "
        f"Model={run_cfg.model.name}, Dataset={run_cfg.dataset.name}, "
        f"Epochs={run_cfg.training.epochs}, Batch={run_cfg.training.batch_size}"
    )
    print(description)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optuna search or single run
    if run_cfg.optuna.n_trials > 0:
        study = optuna.create_study(direction=run_cfg.optuna.direction)
        objective = _objective_factory(run_cfg, results_dir, device)
        study.optimize(objective, n_trials=run_cfg.optuna.n_trials)
        best_value = study.best_value
        best_params = study.best_params
        print(f"Best Optuna value={best_value}, params={best_params}")
        # Merge best params back into config
        for k, v in best_params.items():
            if k in run_cfg.training:
                run_cfg.training[k] = v
            elif k.startswith("dataset_") and hasattr(run_cfg.dataset, k[8:]):
                setattr(run_cfg.dataset, k[8:], v)

    # Final training with best / given hyper-parameters
    final_results = _run_training(run_cfg, results_dir, device)

    # Save results
    _save_json(final_results, results_dir / "results.json")

    # Print JSON-formatted results to STDOUT
    print(json.dumps(final_results))


if __name__ == "__main__":
    train_app()