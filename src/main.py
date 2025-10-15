import json
import subprocess
import sys
from pathlib import Path
import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    # Resolve absolute results directory
    results_dir = Path(cfg.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Launch training as subprocess (ensures clean Hydra context)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"results_dir={results_dir}",
    ]
    if cfg.trial_mode:
        cmd.append("trial_mode=true")
    print(f"Running command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        sys.exit(proc.returncode)

    # After training, trigger evaluation tool across all results
    eval_cmd = [sys.executable, "-m", "src.evaluate", str(results_dir)]
    subprocess.run(eval_cmd)


if __name__ == "__main__":
    main()
