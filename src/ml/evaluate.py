import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml.constants import ML_RESULTS_DIR, PREPROCESS_RESULTS_DIR
from ml.train_models import (
    evaluate_model,
    split_features_targets,
)


def get_evaluation_dir(model_path):
    """Resolve evaluation output directory inside the dataset results folder."""
    model_dir = Path(model_path).parent
    if model_dir.name in {"train_models", "hp_tuning", "evaluation", "finetune"}:
        return model_dir.parent / "evaluation"

    return model_dir / "evaluation"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a saved pkl model on train+test processed data."
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
        help=f"Path to saved .pkl model.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Directory to save evaluation metrics.",
    )

    args = parser.parse_args()
    args.data_dir = args.data_dir_option or args.data_dir
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
        return saved_object["model"], saved_object.get("preprocessing", {})

    return saved_object, {}


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


def save_metrics(metrics, target_metrics, output_dir, dataset_name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"saved_model_test_{dataset_name}_metrics.csv"
    target_metrics_path = output_dir / f"saved_model_test_{dataset_name}_target_metrics.csv"

    pd.DataFrame(
        [
            {
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            }
        ]
    ).to_csv(summary_path, index=False)
    target_metrics.to_csv(target_metrics_path, index=False)

    return summary_path, target_metrics_path


def main():
    args = parse_args()
    model, preprocessing = load_model(args.model_path)
    data = load_train_test_data(args.data_dir)
    features, targets = split_features_targets(data)
    expected_feature_names = get_expected_feature_names(model, preprocessing)
    features = align_features(features, expected_feature_names)

    metrics = evaluate_model(model, features, targets)
    target_metrics = metrics_by_target(metrics)
    tested_dataset_name = Path(args.data_dir).name
    summary_path, target_metrics_path = save_metrics(
        metrics,
        target_metrics,
        args.output_dir,
        tested_dataset_name,
    )

    print("Saved model test metrics:")
    print(f"RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}, R2={metrics['r2']:.6f}")
    print(f"Loaded model from {args.model_path}")
    print(f"Loaded train+test data from {args.data_dir}")
    print(f"Saved summary metrics to {summary_path}")
    print(f"Saved target metrics to {target_metrics_path}")


if __name__ == "__main__":
    main()
