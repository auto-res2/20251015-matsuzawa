"""Main orchestrator launching train subprocess and optional evaluation."""
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base=None)
def _main(cfg: DictConfig) -> None:  # pylint: disable=too-many-locals
    # Get run_id from the loaded config (merged from run/ yaml)
    run_id = cfg.get("run_id", "unknown")
    results_dir = Path(hydra.utils.to_absolute_path(cfg.results_dir)).as_posix()

    # ------------------------------------------------------------------
    # Build subprocess command for training
    # ------------------------------------------------------------------
    cmd: List[str] = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"results_dir={results_dir}",
    ]
    # Pass the run override from command line if present
    for arg in sys.argv[1:]:
        if arg.startswith("run="):
            cmd.append(arg)
            break
    
    # Propagate flags
    if cfg.trial_mode:
        cmd.append("trial_mode=true")
    # Forward wandb override if present
    if "wandb" in cfg and "mode" in cfg.wandb:
        cmd.append(f"wandb.mode={cfg.wandb.mode}")

    print("Launching training subprocess: \n" + " ".join(cmd))
    subprocess.run(cmd, check=True)

    # ------------------------------------------------------------------
    # After training, launch evaluation (across available results)
    # ------------------------------------------------------------------
    eval_cmd: List[str] = [
        sys.executable,
        "-u",
        "-m",
        "src.evaluate",
        f"results_dir={results_dir}",
        f"wandb.mode={cfg.wandb.mode}",
    ]
    print("Launching evaluation subprocess: \n" + " ".join(eval_cmd))
    subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    _main()
