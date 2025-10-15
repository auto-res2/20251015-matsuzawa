import json
import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    results_root = Path(to_absolute_path(cfg.results_dir))
    results_root.mkdir(parents=True, exist_ok=True)

    def _run_single(run_id: str):
        run_results_dir = results_root / run_id
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "src.train",
            f"--config-name={run_id}",
            f"results_dir={run_results_dir}",
            f"trial_mode={cfg.trial_mode}",
            f"wandb.mode={cfg.wandb.mode}",
        ]
        print("Executing: ", " ".join(map(str, cmd)))
        subprocess.run(cmd, check=True)

    # Determine which runs to execute------------------------------------------------
    run_list: List[str]
    run_param = cfg.get("run", "all")
    if run_param == "all":
        run_list = cfg.run_list
    else:
        run_list = [run_param]
    for r in run_list:
        _run_single(r)

    # After all runs, trigger evaluation ------------------------------------------------
    cmd_eval = [
        sys.executable,
        "-u",
        "-m",
        "src.evaluate",
        f"results_dir={results_root}",
        f"wandb.mode={cfg.wandb.mode}",
    ]
    print("Executing evaluation: ", " ".join(map(str, cmd_eval)))
    subprocess.run(cmd_eval, check=True)


if __name__ == "__main__":
    main()
