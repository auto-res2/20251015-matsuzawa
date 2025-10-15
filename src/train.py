import json, os, sys, time, random, math, tempfile
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
import hydra
from sklearn.metrics import accuracy_score, f1_score

from .preprocess import build_dataloaders
from .model import build_model, model_num_parameters

try:
    import wandb  # noqa: F401
except ImportError:  # pragma: no cover
    wandb = None


class _DummyWandB:  # pylint: disable=too-few-public-methods
    """A no-op replacement when wandb is disabled."""

    def __getattr__(self, name):
        def _noop(*_, **__):
            return None

        return _noop


# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cfg: DictConfig) -> torch.device:
    if torch.cuda.is_available() and cfg.training.device != "cpu":
        return torch.device("cuda")
    return torch.device("cpu")


def log_experiment_description(cfg: DictConfig):
    description = (
        f"Running experiment '{cfg.run_id}' with method '{cfg.method}'.\n"
        f"Model: {cfg.model.name} | Dataset: {cfg.dataset.name}\n"
        f"Training for {cfg.training.epochs} epochs, batch_size={cfg.training.batch_size}."
    )
    print("=" * 80)
    print(description)
    print("=" * 80, flush=True)


# ----------------------------------------------------------------------------
# Training / Evaluation helpers
# ----------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    loader: torch.utils.data.DataLoader,
) -> Tuple[float, float]:
    model.train()
    epoch_loss, preds, gts = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x.size(0)
        preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
        gts.extend(y.detach().cpu().numpy())
    epoch_loss /= len(loader.dataset)
    return epoch_loss, accuracy_score(gts, preds)


def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    loader: torch.utils.data.DataLoader,
) -> Tuple[float, float, float]:
    model.eval()
    loss, preds, gts = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss += criterion(outputs, y).item() * x.size(0)
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            gts.extend(y.cpu().numpy())
    loss /= len(loader.dataset)
    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds, average="macro")
    return loss, acc, f1


def measure_inference_time(model: nn.Module, device: torch.device, sample: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        sample = sample.to(device)
        # Warm-up
        for _ in range(5):
            _ = model(sample)
        t0 = time.perf_counter()
        _ = model(sample)
        t1 = time.perf_counter()
    return (t1 - t0) * 1000  # ms


# ----------------------------------------------------------------------------
# Trainer entry point
# ----------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config", version_base=None)
def _main(cfg: DictConfig) -> None:  # pylint: disable=too-many-locals
    # Extract run config
    run_cfg = cfg.run
    
    # Apply trial-mode overrides (epochs=1, no Optuna)
    if cfg.trial_mode:
        OmegaConf.set_struct(run_cfg, False)
        run_cfg.training.epochs = 1
        run_cfg.optuna.n_trials = 0
        OmegaConf.set_struct(run_cfg, True)

    # Create results directory structure
    results_root = Path(hydra.utils.to_absolute_path(cfg.results_dir))
    run_dir = results_root / run_cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Logging & description
    # ---------------------------------------------------------------------
    log_experiment_description(run_cfg)

    # ---------------------------------------------------------------------
    # Seed / device
    # ---------------------------------------------------------------------
    set_seed(run_cfg.training.seed)
    device = get_device(run_cfg)

    # ---------------------------------------------------------------------
    # Data
    # ---------------------------------------------------------------------
    train_loader, val_loader, sample_batch = build_dataloaders(run_cfg)

    # ---------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------
    model = build_model(run_cfg.model, run_cfg)
    num_params = model_num_parameters(model)
    model.to(device)

    # ---------------------------------------------------------------------
    # Optimizer & Criterion
    # ---------------------------------------------------------------------
    if run_cfg.training.optimizer.name.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=run_cfg.training.optimizer.lr,
            momentum=run_cfg.training.optimizer.momentum,
            weight_decay=run_cfg.training.optimizer.weight_decay,
        )
    elif run_cfg.training.optimizer.name.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=run_cfg.training.optimizer.lr,
            weight_decay=run_cfg.training.optimizer.weight_decay,
        )
    elif run_cfg.training.optimizer.name.lower() in {"adamw", "adam_w"}:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=run_cfg.training.optimizer.lr,
            weight_decay=run_cfg.training.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer {run_cfg.training.optimizer.name}")

    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------------------------
    # WandB initialisation
    # ---------------------------------------------------------------------
    wb_mode = cfg.wandb.mode.lower()
    use_wandb = wb_mode != "disabled" and wandb is not None
    wb_run = _DummyWandB()  # type: ignore
    if use_wandb:
        wb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(run_cfg, resolve=True),
            reinit=True,
            mode=wb_mode,
            name=run_cfg.run_id,
        )
        # Save WandB metadata for later GI actions
        metadata = {
            "wandb_entity": cfg.wandb.entity,
            "wandb_project": cfg.wandb.project,
            "wandb_run_id": wb_run.id,
        }
        with (run_dir / "wandb_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"WandB URL: {wb_run.url}")

    # ---------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}

    best_val_acc = 0.0
    for epoch in range(1, run_cfg.training.epochs + 1):
        t_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, device, train_loader)
        val_loss, val_acc, val_f1 = evaluate(model, criterion, device, val_loader)
        t_end = time.perf_counter()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        # WandB logging
        if use_wandb:
            wb_run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "epoch_time": t_end - t_start,
                }
            )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = run_dir / "best_model.pt"
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, ckpt_path)
            if use_wandb:
                wb_run.save(str(ckpt_path))

    # ---------------------------------------------------------------------
    # Final metrics
    # ---------------------------------------------------------------------
    inf_time = measure_inference_time(model, device, sample_batch)

    results: Dict[str, Any] = {
        "run_id": run_cfg.run_id,
        "method": run_cfg.method,
        "dataset": run_cfg.dataset.name,
        "model": run_cfg.model.name,
        "final_val_accuracy": history["val_acc"][-1],
        "final_val_f1": history["val_f1"][-1],
        "inference_time_ms": inf_time,
        "model_num_params": num_params,
        "history": history,
    }

    # Output experimental numerical data to stdout
    print(json.dumps(results, indent=2))

    # Save to file
    with (run_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if use_wandb:
        wb_run.finish()


if __name__ == "__main__":
    _main()
