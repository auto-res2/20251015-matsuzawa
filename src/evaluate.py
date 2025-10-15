"""src/evaluate.py
Collects all single-run result files and produces comparison plots.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def _main(cfg: DictConfig):
    results_dir = Path(hydra.utils.get_original_cwd()) / cfg.results_dir
    run_ids: List[str] = cfg.available_runs

    summaries: Dict[str, Dict] = {}

    for run_id in run_ids:
        res_file = results_dir / run_id / "results.json"
        if not res_file.exists():
            print(f"Warning – results for {run_id} not found. Skipping.")
            continue
        with open(res_file, "r", encoding="utf-8") as fp:
            summaries[run_id] = json.load(fp)

    # Output comparison table ---------------------------------------------------
    print("\n===== Aggregate Results =====")
    for run_id, summary in summaries.items():
        print(f"{run_id}: {summary['metric_name']} = {summary['best_val_metric']:.4f}")

    # Bar plot ------------------------------------------------------------------
    labels = list(summaries.keys())
    values = [summaries[r]["best_val_metric"] for r in labels]

    plt.figure(figsize=(10, 4))
    plt.barh(labels, values)
    plt.xlabel("Best validation metric")
    plt.title("Comparison across runs")
    plt.tight_layout()
    fig_path = results_dir / "comparison.png"
    plt.savefig(fig_path)

    # WandB artifact -------------------------------------------------------------
    if cfg.wandb.mode in ["online", "offline"]:
        import wandb

        run = wandb.init(entity="gengaru617", project="251015-test", name="evaluation", mode=cfg.wandb.mode)
        run.log({"comparison": wandb.Image(str(fig_path))})
        run.finish()

    # Print final comparison JSON ----------------------------------------------
    print(json.dumps({"comparison": summaries}))


if __name__ == "__main__":
    _main()
