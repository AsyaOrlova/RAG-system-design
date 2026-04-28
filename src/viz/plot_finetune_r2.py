import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FINETUNE_DIR = PROJECT_ROOT / "results" / "complexes" / "finetune"

METRICS_PATTERN = re.compile(
    r"^finetuned_model_test_(?P<train_size>\d+(?:\.\d+)?)_(?P<dataset>.+?)_metrics\.csv$"
)
TARGET_METRICS_SUFFIX = "_target_metrics.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot mean R2 vs finetune train_size from finetune metrics."
    )
    parser.add_argument(
        "finetune_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_FINETUNE_DIR,
        help=f"Directory with finetune metrics CSV files. Defaults to {DEFAULT_FINETUNE_DIR}",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        help="Path to output image. Defaults to <finetune_dir>/r2_vs_train_size.png",
    )
    return parser.parse_args()


def load_points(finetune_dir):
    finetune_dir = Path(finetune_dir)
    rows = []

    for metrics_path in sorted(finetune_dir.glob("finetuned_model_test_*_metrics.csv")):
        if metrics_path.name.endswith(TARGET_METRICS_SUFFIX):
            continue

        match = METRICS_PATTERN.match(metrics_path.name)
        if not match:
            continue

        metrics = pd.read_csv(metrics_path)
        if "r2" not in metrics.columns or metrics.empty:
            continue

        rows.append(
            {
                "train_size": float(match.group("train_size")),
                "dataset": match.group("dataset"),
                "r2": float(metrics.iloc[0]["r2"]),
            }
        )

    if not rows:
        raise ValueError(f"No finetune metrics files found in {finetune_dir}")

    return pd.DataFrame(rows).sort_values("train_size").reset_index(drop=True)


def build_plot(points_df, output_path, title):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 6))
    x_values = points_df["train_size"].tolist()
    x_positions = list(range(len(x_values)))
    x_labels = [f"{value:g}" for value in x_values]

    ax.plot(
        x_positions,
        points_df["r2"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        color="#2563eb",
    )

    for x_position, row in zip(x_positions, points_df.itertuples(index=False)):
        ax.annotate(
            f"{row.r2:.4f}",
            (x_position, row.r2),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    ax.set_title(title)
    ax.set_xlabel("train_size")
    ax.set_ylabel("mean R2")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_positions, x_labels)
    ax.set_xlim(-0.2, len(x_positions) - 0.8 if len(x_positions) > 1 else 0.2)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    points_df = load_points(args.finetune_dir)
    output_path = args.output_path or Path(args.finetune_dir) / "r2_vs_train_size.png"
    title = f"Mean R2 vs train_size ({Path(args.finetune_dir).name})"

    build_plot(points_df, output_path, title)

    print(f"Loaded {len(points_df)} point(s) from {args.finetune_dir}")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
