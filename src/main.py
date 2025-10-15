import os
import subprocess
from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import to_absolute_path


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main_app(cfg: DictConfig):
    """Main orchestrator: launches training as subprocess & optional evaluation."""
    run_ids = [cfg.run] if isinstance(cfg.run, str) else cfg.run
    if run_ids is None:
        raise ValueError("No run id provided. Usage: python -m src.main run=<run_id>")

    for run_id in run_ids:
        cmd = [
            "python",
            "-u",
            "-m",
            "src.train",
            f"run={run_id}",
            f"results_dir={cfg.results_dir}",
            f"trial_mode={str(cfg.trial_mode).lower()}",
        ]
        print("Launching:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    # After all runs complete, aggregate results
    if cfg.get("evaluate", True):
        cmd_eval = [
            "python",
            "-u",
            "-m",
            "src.evaluate",
            f"results_dir={cfg.results_dir}",
        ]
        subprocess.run(cmd_eval, check=True)


if __name__ == "__main__":
    main_app()