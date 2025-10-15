import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import hydra
from omegaconf import DictConfig

try:
    import wandb  # noqa: F401
except ImportError:
    wandb = None  # pragma: no cover


@hydra.main(config_path="../config", config_name="config", version_base=None)
def _main(cfg: DictConfig) -> None:  # pylint: disable=too-many-locals
    results_root = Path(hydra.utils.to_absolute_path(cfg.results_dir))
    result_files = list(results_root.glob("*/results.json"))
    if not result_files:
        print("No result files found.")
        sys.exit(0)

    summary: List[Dict] = []
    for rf in result_files:
        with rf.open("r", encoding="utf-8") as f:
            summary.append(json.load(f))

    # Prepare plots
    run_ids = [r["run_id"] for r in summary]
    accuracies = [r["final_val_accuracy"] for r in summary]
    f1s = [r["final_val_f1"] for r in summary]
    inference = [r["inference_time_ms"] for r in summary]
    params = [r["model_num_params"] for r in summary]

    x = np.arange(len(run_ids))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, accuracies, width, label="Accuracy")
    bars2 = ax.bar(x, f1s, width, label="F1")
    bars3 = ax.bar(x + width, inference, width, label="Inference(ms)")

    ax.set_xticks(x)
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.legend()
    ax.set_title("Comparison of Metrics across Runs")
    fig.tight_layout()

    fig_path = results_root / "comparison.png"
    fig.savefig(fig_path)

    # WandB upload if enabled
    use_wandb = cfg.wandb.mode.lower() != "disabled" and wandb is not None
    if use_wandb:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            mode=cfg.wandb.mode,
            reinit=True,
            name="evaluation-comparison",
        )
        wandb_run.log({"comparison": wandb.Image(str(fig_path))})
        wandb_run.finish()

    # Print aggregated results
    comparison = {
        "run_ids": run_ids,
        "accuracy": accuracies,
        "f1": f1s,
        "inference_time_ms": inference,
        "params": params,
    }
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    _main()
