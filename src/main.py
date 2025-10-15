import os
import subprocess
from pathlib import Path
import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import to_absolute_path


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main_app(cfg: DictConfig):
    """Main orchestrator: launches training as subprocess & optional evaluation."""
    run_id = cfg.run_id if hasattr(cfg, 'run_id') else cfg.run.run_id
    
    cmd = [
        "python",
        "-u",
        "-m",
        "src.train",
        f"run={run_id}",
        f"results_dir={cfg.results_dir}",
        f"trial_mode={str(cfg.trial_mode).lower()}",
        f"wandb.mode={cfg.wandb.mode}",
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