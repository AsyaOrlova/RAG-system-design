import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .evaluate_top_hits import evaluate_experiment
except ImportError:
    from evaluate_top_hits import evaluate_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS_DIR = PROJECT_ROOT / "optimization" / "experiments"
DEFAULT_OUTPUT_NAME = "experiments_summary.csv"
PARAMETER_COLUMNS = [
    "chunk_size",
    "chunk_overlap",
    "dense_k",
    "sparse_k",
    "search_mode",
    "reranker",
    "llm_model",
]


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def infer_source_from_model_path(model_path):
    parts = Path(model_path).parts
    if "results" in parts:
        index = parts.index("results") + 1
        if index < len(parts):
            return parts[index]

    return ""


def add_top_hit_columns(row, experiment_dir):
    try:
        top_hit_rows = evaluate_experiment(experiment_dir)
    except Exception as exc:
        row["top_hit_error"] = f"{type(exc).__name__}: {exc}"
        return row

    row["top_hit_error"] = ""
    for hit_row in top_hit_rows:
        prefix = f"{hit_row['top_list']}_"
        row[f"{prefix}rank"] = hit_row["rank"]
        row[f"{prefix}in_top_1"] = hit_row["in_top_1"]
        row[f"{prefix}in_top_3"] = hit_row["in_top_3"]
        row[f"{prefix}in_top_5"] = hit_row["in_top_5"]
        row[f"{prefix}in_top_10"] = hit_row["in_top_10"]

    return row


def build_experiment_row(experiment_dir):
    best_parameters_path = experiment_dir / "best_parameters.json"
    best_parameters = load_json(best_parameters_path)
    params = best_parameters.get("params", {})
    predicted_metrics = {
        metric: value
        for metric, value in best_parameters.get("predicted_metrics", {}).items()
        if not metric.startswith("param_")
    }
    model_path = best_parameters.get("model_path", "")

    row = {
        "experiment": experiment_dir.name,
        "target_dataset": best_parameters.get("target_dataset", ""),
        "source_dataset": infer_source_from_model_path(model_path),
        "model_path": model_path,
        "objective_value": best_parameters.get("objective_value", ""),
        "objective_metrics": ",".join(best_parameters.get("objective_metrics", [])),
    }

    for parameter in PARAMETER_COLUMNS:
        row[parameter] = params.get(parameter, "")

    for metric, value in predicted_metrics.items():
        row[f"predicted_{metric}"] = value

    return add_top_hit_columns(row, experiment_dir)


def summarize_experiments(experiments_dir):
    experiments_dir = Path(experiments_dir)
    rows = []

    for experiment_dir in sorted(path for path in experiments_dir.iterdir() if path.is_dir()):
        best_parameters_path = experiment_dir / "best_parameters.json"
        if not best_parameters_path.exists():
            continue

        rows.append(build_experiment_row(experiment_dir))

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect all optimization experiments into one CSV table."
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=DEFAULT_EXPERIMENTS_DIR,
        help=f"Directory with optimization experiments. Defaults to {DEFAULT_EXPERIMENTS_DIR}",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        help="Output CSV path. Defaults to <experiments-dir>/experiments_summary.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = args.output_path or args.experiments_dir / DEFAULT_OUTPUT_NAME
    summary = summarize_experiments(args.experiments_dir)
    summary.to_csv(output_path, index=False)

    print(f"Saved {len(summary)} experiment rows to {output_path}")


if __name__ == "__main__":
    main()
