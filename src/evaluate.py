import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_results(results_dir: Path) -> List[Dict]:
    results = []
    for p in results_dir.glob("*/results.json"):
        with p.open() as f:
            obj = json.load(f)
            final = obj["final_metrics"]
            final["run_id"] = p.parent.name
            results.append(final)
    return results


def main(results_dir: str):
    results_path = Path(results_dir)
    records = load_results(results_path)
    if not records:
        print("No result files found", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records).set_index("run_id")
    # Plot accuracy comparison
    ax = df["test_accuracy"].plot(kind="bar", figsize=(10, 4), ylabel="Test Accuracy")
    fig = ax.get_figure()
    fig.tight_layout()
    fig_path = results_path / "accuracy_comparison.png"
    fig.savefig(fig_path)

    summary = {
        "best_run": df["test_accuracy"].idxmax(),
        "best_accuracy": df["test_accuracy"].max(),
        "all_runs": records,
    }

    # Print JSON summary
    print(json.dumps(summary, indent=2))

    # WandB artifact upload if metadata exists
    meta_file = results_path / summary["best_run"] / "wandb_metadata.json"
    if meta_file.exists():
        try:
            import wandb
            with meta_file.open() as f:
                meta = json.load(f)
            run = wandb.init(
                project=meta["wandb_project"],
                entity=meta["wandb_entity"],
                id=meta["wandb_run_id"],
                resume="allow",
                reinit=True,
                mode="online",
            )
            run.log({"accuracy_comparison": wandb.Image(str(fig_path))})
            run.finish()
        except Exception as e:
            print(f"wandb upload failed: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.evaluate <results_dir>")
        sys.exit(1)
    main(sys.argv[1])
