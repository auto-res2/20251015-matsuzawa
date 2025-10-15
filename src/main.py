import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import List

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


def tee_stream(stream, log_file_path):
    """Read from a stream, write simultaneously to stdout and a file."""
    with open(log_file_path, "w", encoding="utf-8") as log_fp:
        for line in iter(stream.readline, b""):
            decoded = line.decode()
            print(decoded, end="")
            log_fp.write(decoded)
    stream.close()


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_id = cfg.run
    if run_id is None:
        print("Error: run parameter must be provided.")
        sys.exit(1)

    # Launch train.py as subprocess
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.train",
        f"run={run_id}",
        f"results_dir={results_dir}",
    ]
    if cfg.get("trial_mode", False):
        cmd.append("trial_mode=true")

    print(f"Launching experiment {run_id} with command: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Tee stdout/stderr
    stdout_log = results_dir / run_id / "stdout.log"
    stderr_log = results_dir / run_id / "stderr.log"
    stdout_thread = threading.Thread(target=tee_stream, args=(proc.stdout, stdout_log))
    stderr_thread = threading.Thread(target=tee_stream, args=(proc.stderr, stderr_log))
    stdout_thread.start()
    stderr_thread.start()
    proc.wait()
    stdout_thread.join()
    stderr_thread.join()
    if proc.returncode != 0:
        print(f"Experiment {run_id} failed with exit code {proc.returncode}")
        sys.exit(proc.returncode)

    # After training, run evaluation across all runs if desired
    evaluate_cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.evaluate",
        str(results_dir),
        "--run_ids",
        run_id,
    ]
    subprocess.run(evaluate_cmd, check=True)


if __name__ == "__main__":
    main()
