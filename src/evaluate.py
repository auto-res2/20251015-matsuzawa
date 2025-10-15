import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import wandb


def gather_results(results_dir: Path, run_ids: List[str]) -> Dict[str, Dict]:
    all_results = {}
    for rid in run_ids:
        result_path = results_dir / rid / "results.json"
        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as fp:
                all_results[rid] = json.load(fp)
        else:
            print(f"Warning: results for {rid} not found at {result_path}")
    return all_results


def plot_comparison(all_results: Dict[str, Dict], save_path: Path):
    labels = list(all_results.keys())
    accuracies = [res["best_val_accuracy"] for res in all_results.values()]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, accuracies)
    plt.ylabel("Best Validation Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main(results_dir: str, run_ids: List[str]):
    results_dir = Path(results_dir)
    all_results = gather_results(results_dir, run_ids)

    # Print aggregated JSON
    print(json.dumps(all_results, indent=2))

    # Plot comparison figure
    fig_path = results_dir / "comparison.png"
    plot_comparison(all_results, fig_path)

    # Upload to WandB as artifact
    wandb_run = wandb.init(project="251015-test", entity="gengaru617", name="evaluation", job_type="evaluation")
    wandb_run.log({"comparison_plot": wandb.Image(str(fig_path))})
    wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=str, help="Path where result folders live")
    parser.add_argument("--run_ids", nargs="*", required=True)
    args = parser.parse_args()
    main(args.results_dir, args.run_ids)
