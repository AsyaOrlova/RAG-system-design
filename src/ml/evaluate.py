import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml.train_models import (
    evaluate_model,
    split_features_targets,
)


def get_evaluation_dir(model_path):
    """Resolve evaluation output directory inside the dataset results folder."""
    model_path = Path(model_path)
    if model_path.is_dir():
        if model_path.name == "best_tuned_model_runs":
            return model_path.parent.parent / "evaluation"
        return model_path / "evaluation"

    model_dir = Path(model_path).parent
    if model_dir.name == "best_tuned_model_runs":
        return model_dir.parent.parent / "evaluation"

    if model_dir.name in {"train_models", "hp_tuning", "evaluation", "finetune"}:
        return model_dir.parent / "evaluation"

    return model_dir / "evaluation"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test saved pkl model(s) on processed test.csv data."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        help="Path to a processed dataset directory with train.csv and test.csv.",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        dest="data_dir_option",
        type=Path,
        help=f"Path to processed train/test data.",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=Path,
        required=True,
        help=(
            "Path to a saved .pkl model or to a directory with saved .pkl models. "
            "If best_tuned_model.pkl is passed and best_tuned_model_runs exists "
            "next to it, all run models are evaluated."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Directory to save evaluation metrics.",
    )

    args = parser.parse_args()
    args.data_dir = args.data_dir_option or args.data_dir
    if args.data_dir is None:
        parser.error("data_dir must be provided either positionally or with -d")
    args.output_dir = args.output_dir or get_evaluation_dir(args.model_path)
    del args.data_dir_option
    return args


def load_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model was not found: {model_path}")

    with model_path.open("rb") as model_file:
        saved_object = pickle.load(model_file)

    if isinstance(saved_object, dict) and "model" in saved_object:
        preprocessing = saved_object.get("preprocessing", {}).copy()
        if "run" in saved_object:
            preprocessing["run"] = saved_object["run"]
        if "random_state" in saved_object:
            preprocessing["random_state"] = saved_object["random_state"]
        return saved_object["model"], preprocessing

    return saved_object, {}


def resolve_model_paths(model_path):
    def sort_key(path):
        stem_parts = path.stem.split("_")
        last_part = stem_parts[-1] if stem_parts else ""
        return (int(last_part) if last_part.isdigit() else 0, path.name)

    model_path = Path(model_path)
    if model_path.is_dir():
        model_paths = sorted(model_path.glob("*.pkl"), key=sort_key)
        if not model_paths:
            raise FileNotFoundError(f"No .pkl models were found in {model_path}")
        return model_paths

    if not model_path.exists():
        raise FileNotFoundError(f"Saved model was not found: {model_path}")

    sibling_runs_dir = model_path.parent / "best_tuned_model_runs"
    if model_path.name == "best_tuned_model.pkl" and sibling_runs_dir.exists():
        model_paths = sorted(sibling_runs_dir.glob("*.pkl"), key=sort_key)
        if model_paths:
            return model_paths

    if model_path.parent.name == "best_tuned_model_runs":
        model_paths = sorted(model_path.parent.glob("*.pkl"), key=sort_key)
        if model_paths:
            return model_paths

    return [model_path]


def load_train_test_data(data_dir):
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    missing_files = [str(path) for path in [train_path, test_path] if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "Processed train/test files were not found. "
            f"Missing: {missing_files}"
        )

    return pd.concat(
        [pd.read_csv(train_path), pd.read_csv(test_path)],
        axis=0,
        ignore_index=True,
    )


def load_test_data(data_dir):
    data_dir = Path(data_dir)
    test_path = data_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Processed test file was not found: {test_path}")

    return pd.read_csv(test_path)


def get_expected_feature_names(model, preprocessing=None):
    preprocessing = preprocessing or {}
    if preprocessing.get("feature_names"):
        return list(preprocessing["feature_names"])

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "feature_names_"):
        return list(model.feature_names_)

    estimators = getattr(model, "estimators_", None)
    if estimators:
        first_estimator = estimators[0]
        if hasattr(first_estimator, "feature_names_in_"):
            return list(first_estimator.feature_names_in_)
        if hasattr(first_estimator, "feature_names_"):
            return list(first_estimator.feature_names_)

    return None


def align_features(features, expected_feature_names):
    if expected_feature_names is None:
        return features

    aligned = features.copy()
    missing_columns = [
        column for column in expected_feature_names if column not in aligned.columns
    ]

    for column in missing_columns:
        aligned[column] = 0

    return aligned[expected_feature_names]


def metrics_by_target(metrics):
    rows = []

    for target_name, rmse_value in metrics["rmse_per_target"].items():
        rows.append(
            {
                "target": target_name,
                "rmse": rmse_value,
                "mae": metrics["mae_per_target"][target_name],
                "r2": metrics["r2_per_target"][target_name],
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics(metrics_df):
    metric_order = ["r2", "rmse", "mae"]
    metrics_long = metrics_df.melt(
        id_vars=["run", "random_state", "model_path"],
        value_vars=["rmse", "mae", "r2"],
        var_name="metric",
        value_name="value",
    )
    summary = (
        metrics_long.groupby("metric")["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    summary["mean_std"] = summary.apply(
        lambda row: f"{row['mean']:.6f} +- {row['std']:.6f}",
        axis=1,
    )
    summary["metric"] = pd.Categorical(
        summary["metric"],
        categories=metric_order,
        ordered=True,
    )
    summary = summary.sort_values("metric").reset_index(drop=True)
    summary["metric"] = summary["metric"].astype(str)
    return summary[["metric", "mean_std"]]


def summarize_target_metrics(target_metrics_df):
    metric_order = ["r2", "rmse", "mae"]
    metrics_long = target_metrics_df.melt(
        id_vars=["run", "random_state", "model_path", "target"],
        value_vars=["rmse", "mae", "r2"],
        var_name="metric",
        value_name="value",
    )
    summary = (
        metrics_long.groupby(["target", "metric"])["value"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    summary["mean_std"] = summary.apply(
        lambda row: f"{row['mean']:.6f} +- {row['std']:.6f}",
        axis=1,
    )
    summary["metric"] = pd.Categorical(
        summary["metric"],
        categories=metric_order,
        ordered=True,
    )
    summary = summary.sort_values(["metric", "target"]).reset_index(drop=True)
    summary["metric"] = summary["metric"].astype(str)
    return summary[["metric", "target", "mean_std"]]


def evaluate_saved_models(model_paths, features, targets):
    metric_rows = []
    target_metric_frames = []

    for run_index, model_path in enumerate(model_paths, start=1):
        model, preprocessing = load_model(model_path)
        expected_feature_names = get_expected_feature_names(model, preprocessing)
        aligned_features = align_features(features, expected_feature_names)
        metrics = evaluate_model(model, aligned_features, targets)

        random_state = preprocessing.get("random_state")

        metric_rows.append(
            {
                "run": run_index,
                "random_state": random_state,
                "model_path": str(model_path),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            }
        )

        target_metrics = metrics_by_target(metrics)
        target_metrics.insert(0, "model_path", str(model_path))
        target_metrics.insert(0, "random_state", random_state)
        target_metrics.insert(0, "run", run_index)
        target_metric_frames.append(target_metrics)

    metrics_df = pd.DataFrame(metric_rows)
    target_metrics_df = pd.concat(target_metric_frames, ignore_index=True)
    return metrics_df, target_metrics_df


def save_metrics(
    metrics_summary,
    target_metrics_summary,
    run_metrics,
    run_target_metrics,
    output_dir,
    dataset_name,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"saved_model_test_{dataset_name}_metrics.csv"
    target_metrics_path = output_dir / f"saved_model_test_{dataset_name}_target_metrics.csv"
    run_metrics_path = output_dir / f"saved_model_test_{dataset_name}_model_runs.csv"
    run_target_metrics_path = (
        output_dir / f"saved_model_test_{dataset_name}_target_model_runs.csv"
    )

    metrics_summary.to_csv(summary_path, index=False)
    target_metrics_summary.to_csv(target_metrics_path, index=False)
    run_metrics.to_csv(run_metrics_path, index=False)
    run_target_metrics.to_csv(run_target_metrics_path, index=False)

    return summary_path, target_metrics_path, run_metrics_path, run_target_metrics_path


def main():
    args = parse_args()
    model_paths = resolve_model_paths(args.model_path)
    data = load_test_data(args.data_dir)
    features, targets = split_features_targets(data)
    run_metrics, run_target_metrics = evaluate_saved_models(
        model_paths,
        features,
        targets,
    )
    metrics_summary = summarize_metrics(run_metrics)
    target_metrics_summary = summarize_target_metrics(run_target_metrics)
    tested_dataset_name = Path(args.data_dir).name
    (
        summary_path,
        target_metrics_path,
        run_metrics_path,
        run_target_metrics_path,
    ) = save_metrics(
        metrics_summary,
        target_metrics_summary,
        run_metrics,
        run_target_metrics,
        args.output_dir,
        tested_dataset_name,
    )

    print("Saved model test metrics:")
    print(metrics_summary.to_string(index=False))
    print(f"Loaded {len(model_paths)} model(s) from {args.model_path}")
    print(f"Loaded test data from {args.data_dir}")
    print(f"Saved summary metrics to {summary_path}")
    print(f"Saved target metrics to {target_metrics_path}")
    print(f"Saved per-model metrics to {run_metrics_path}")
    print(f"Saved per-model target metrics to {run_target_metrics_path}")


if __name__ == "__main__":
    main()
