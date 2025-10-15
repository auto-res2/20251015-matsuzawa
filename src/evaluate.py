import json
import os
from pathlib import Path
from typing import List, Dict

import hydra
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


class NoOpWandB:
    def __init__(self):
        self.enabled = False

    def init(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def maybe_init_wandb(cfg):
    if cfg.wandb.mode == "disabled":
        return NoOpWandB()
    import wandb

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"evaluation-{cfg.results_dir}",
        mode=cfg.wandb.mode,
    )
    run.enabled = True
    return run


def read_results(dir_path: Path) -> Dict:
    with (dir_path / "results.json").open() as fp:
        return json.load(fp)


def aggregate_results(results_dirs: List[Path]):
    records = [read_results(p) for p in results_dirs]
    return records


def plot_accuracy(records: List[Dict], out_path: Path):
    names = [r["run_id"] for r in records]
    accs = [r["best_val_accuracy"] for r in records]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, accs)
    ax.set_ylabel("Validation Accuracy")
    ax.set_xticklabels(names, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


@hydra.main(config_path="../config", config_name="config")
def main(cfg) -> None:
    results_root = Path(cfg.results_dir)
    sub_dirs = [p for p in results_root.iterdir() if p.is_dir()]
    records = aggregate_results(sub_dirs)

    # Print aggregated numbers --------------------------------------------------
    summary = {r["run_id"]: r["best_val_accuracy"] for r in records}
    print(json.dumps(summary, indent=2))

    # Plot & (optionally) upload to WandB ---------------------------------------
    fig_path = results_root / "comparison.png"
    plot_accuracy(records, fig_path)

    wb = maybe_init_wandb(cfg)
    if getattr(wb, "enabled", False):
        wb.log({"validation_accuracy_comparison": wandb.Image(str(fig_path))})  # type: ignore
        wb.save(str(fig_path))  # type: ignore
        wb.finish()  # type: ignore


if __name__ == "__main__":
    main()
