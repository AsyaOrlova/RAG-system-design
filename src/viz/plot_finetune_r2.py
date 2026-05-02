import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINETUNE_DIR = PROJECT_ROOT / "results" / "complexes" / "finetune"

TARGET_METRICS_PATTERN = re.compile(
    r"^finetuned_model_test_(?P<train_size>\d+(?:\.\d+)?)_(?P<dataset>.+?)_target_metrics\.csv$"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot target-level R2 vs finetune train_size from finetune metrics."
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

    for metrics_path in sorted(
        finetune_dir.glob("finetuned_model_test_*_target_metrics.csv")
    ):
        match = TARGET_METRICS_PATTERN.match(metrics_path.name)
        if not match:
            continue

        metrics = pd.read_csv(metrics_path)
        required_columns = {"target", "r2"}
        if not required_columns.issubset(metrics.columns) or metrics.empty:
            continue

        for row in metrics.itertuples(index=False):
            rows.append(
                {
                    "train_size": float(match.group("train_size")),
                    "dataset": match.group("dataset"),
                    "target": row.target,
                    "r2": float(row.r2),
                }
            )

    if not rows:
        raise ValueError(f"No finetune target metrics files found in {finetune_dir}")

    return (
        pd.DataFrame(rows)
        .sort_values(["target", "train_size"])
        .reset_index(drop=True)
    )


def build_plot(points_df, output_path, title):
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(12, 7))
    x_values = sorted(points_df["train_size"].unique())
    x_positions = list(range(len(x_values)))
    x_position_by_value = dict(zip(x_values, x_positions))
    x_labels = [f"{value:g}" for value in x_values]

    for target, target_points in points_df.groupby("target", sort=True):
        target_points = target_points.sort_values("train_size")
        ax.plot(
            [x_position_by_value[value] for value in target_points["train_size"]],
            target_points["r2"],
            marker="o",
            linewidth=2,
            markersize=5,
            label=target,
        )

    ax.set_title(title)
    ax.set_xlabel("finetune dataset size")
    ax.set_ylabel("R2")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_positions, x_labels)
    ax.set_xlim(-0.2, len(x_positions) - 0.8 if len(x_positions) > 1 else 0.2)
    ax.legend(
        title="metric",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def infer_base_dataset(finetune_dir):
    finetune_dir = Path(finetune_dir)
    if finetune_dir.name == "finetune" and finetune_dir.parent.name:
        return finetune_dir.parent.name

    return finetune_dir.name


def format_dataset_list(datasets):
    return ", ".join(sorted(str(dataset) for dataset in datasets))


def main():
    args = parse_args()
    points_df = load_points(args.finetune_dir)
    output_path = args.output_path or Path(args.finetune_dir) / "r2_vs_train_size.png"
    base_dataset = infer_base_dataset(args.finetune_dir)
    finetune_datasets = format_dataset_list(points_df["dataset"].unique())
    title = (
        f"Model trained on {base_dataset}, finetuned and tested on {finetune_datasets}\n"
        "R2 by metric vs finetune dataset size"
    )

    build_plot(points_df, output_path, title)

    print(
        f"Loaded {len(points_df)} point(s) for "
        f"{points_df['target'].nunique()} metric(s) from {args.finetune_dir}"
    )
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
