"""src/main.py
Main orchestrator: launches train.py subprocess for a given run_id and finally
dispatches evaluation.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def _main(cfg: DictConfig):
    original_cwd = Path(hydra.utils.get_original_cwd())
    python = sys.executable

    run_ids: List[str] = [cfg.run] if cfg.get("run") else cfg.available_runs

    for run_id in run_ids:
        print(f"\n===== Launching run {run_id} =====")
        # Build command
        cmd = [
            python,
            "-u",
            "-m",
            "src.train",
            f"run={run_id}",
            f"results_dir={cfg.results_dir}",
            f"trial_mode={str(cfg.trial_mode).lower()}",
            f"wandb.mode={cfg.wandb.mode}",
        ]
        print("Executing:", " ".join(cmd))
        res = subprocess.run(cmd, cwd=original_cwd)
        if res.returncode != 0:
            raise SystemExit(f"Run {run_id} failed with exit code {res.returncode}")

    # After all single runs -> evaluation
    print("\nAll runs finished. Launching evaluation...")
    eval_cmd = [
        python,
        "-u",
        "-m",
        "src.evaluate",
        f"results_dir={cfg.results_dir}",
        f"wandb.mode={cfg.wandb.mode}",
    ]
    subprocess.run(eval_cmd, cwd=original_cwd, check=True)


if __name__ == "__main__":
    _main()
