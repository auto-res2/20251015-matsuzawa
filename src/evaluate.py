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
            data = json.load(f)
            summary.append(data)

    # Prepare plots - handle different schemas
    run_ids = []
    accuracies = []
    f1s = []
    inference = []
    params = []
    
    for r in summary:
        run_ids.append(r["run_id"])
        
        # Handle different accuracy field names
        if "final_val_accuracy" in r:
            accuracies.append(r["final_val_accuracy"])
        elif "best_val_accuracy" in r:
            accuracies.append(r["best_val_accuracy"])
        else:
            accuracies.append(0.0)
        
        # Handle different f1 field names
        if "final_val_f1" in r:
            f1s.append(r["final_val_f1"])
        else:
            f1s.append(0.0)
        
        # Handle different inference time field names
        if "inference_time_ms" in r:
            inference.append(r["inference_time_ms"])
        elif "history" in r and len(r["history"]) > 0:
            if isinstance(r["history"], list):
                inference.append(r["history"][-1].get("val_inference_time", 0.0) * 1000)
            else:
                inference.append(0.0)
        else:
            inference.append(0.0)
        
        # Handle different params field names
        if "model_num_params" in r:
            params.append(r["model_num_params"])
        elif "model_parameters" in r:
            params.append(r["model_parameters"])
        else:
            params.append(0)

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
