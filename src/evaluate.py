import json
import os
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb


def _load_results(results_dir):
    result_files = glob(os.path.join(results_dir, "*/results.json"))
    data = []
    for fp in result_files:
        with open(fp, "r", encoding="utf-8") as f:
            data.append(json.load(f))
    return pd.DataFrame(data)


def _plot(df, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for _, row in df.iterrows():
        ax.scatter(row["model_size_mb"], row["val_accuracy"], label=row["run_id"])
    ax.set_xlabel("Model size (MB)")
    ax.set_ylabel("Validation Accuracy")
    ax.legend(fontsize=6)
    ax.set_title("Accuracy vs Model Size")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def evaluate_app(results_dir):
    results_dir = Path(results_dir)
    df = _load_results(results_dir)
    if df.empty:
        print("No results found in", results_dir)
        return
    best_row = df.sort_values("val_accuracy", ascending=False).iloc[0]
    comparison = {
        "best_run_id": best_row["run_id"],
        "best_val_accuracy": best_row["val_accuracy"],
        "runs": df.to_dict(orient="records"),
    }
    # Plot
    plot_path = results_dir / "accuracy_vs_model_size.png"
    _plot(df, plot_path)

    # Optionally log to WandB if metadata exists
    md_path = results_dir / "wandb_metadata.json"
    if md_path.exists():
        with open(md_path, "r", encoding="utf-8") as f:
            md = json.load(f)
        wandb_run = wandb.init(
            entity=md["wandb_entity"],
            project=md["wandb_project"],
            id=md["wandb_run_id"],
            resume="allow",
        )
        wandb_run.log({"comparison_plot": wandb.Image(str(plot_path))})
        wandb_run.finish()

    # Print JSON to STDOUT
    print(json.dumps(comparison))


if __name__ == "__main__":
    # Parse results_dir from command line arguments
    results_dir = "./results"
    for arg in sys.argv[1:]:
        if arg.startswith("results_dir="):
            results_dir = arg.split("=", 1)[1]
    evaluate_app(results_dir)