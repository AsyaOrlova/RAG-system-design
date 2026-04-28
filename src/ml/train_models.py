import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml.constants import (
    TARGET_COLUMNS,
    ML_RESULTS_DIR,
    TRAIN_MODELS_RANDOM_STATE
)

def split_features_targets(data, target_columns=None):
    """Split a processed dataframe into feature and target matrices."""
    target_columns = target_columns or TARGET_COLUMNS
    missing_targets = [column for column in target_columns if column not in data.columns]

    if missing_targets:
        raise ValueError(f"Missing target columns in processed data: {missing_targets}")

    return data.drop(columns=target_columns), data[target_columns]


def resolve_data_dirs(data_dirs=None):
    """Normalize one or more processed dataset directories."""

    if isinstance(data_dirs, (str, Path)):
        return [Path(data_dirs)]

    return [Path(data_dir) for data_dir in data_dirs]


def load_processed_frames(data_dir):
    """Load raw processed train/test dataframes from one directory."""
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    missing_files = [str(path) for path in [train_path, test_path] if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "Processed train/test files were not found. Run src/preprocess.py first. "
            f"Missing: {missing_files}"
        )

    return pd.read_csv(train_path), pd.read_csv(test_path)


def load_processed_data(data_dirs, return_metadata=False):
    """Load train/test files from one or more processed dataset directories."""
    data_dirs = resolve_data_dirs(data_dirs)
    train_frames = []
    test_frames = []

    for data_dir in data_dirs:
        train, test = load_processed_frames(data_dir)

        train_frames.append(train)
        test_frames.append(test)

    train = pd.concat(train_frames, axis=0, ignore_index=True, sort=False)
    test = pd.concat(test_frames, axis=0, ignore_index=True, sort=False)
    x_train, y_train = split_features_targets(train)
    x_test, y_test = split_features_targets(test)
    x_train, x_test = x_train.align(x_test, join="outer", axis=1, fill_value=0)

    if return_metadata:
        return (
            x_train,
            x_test,
            y_train,
            y_test,
            {
                "feature_names": x_train.columns.tolist(),
            },
        )

    return x_train, x_test, y_train, y_test


def load_transfer_data(train_data_dirs, test_data_dirs, return_metadata=False):
    """Load source train files for training and target test files for testing."""
    train_data_dirs = resolve_data_dirs(train_data_dirs)
    test_data_dirs = resolve_data_dirs(test_data_dirs)

    train_frames = []
    test_frames = []

    for data_dir in train_data_dirs:
        train, _ = load_processed_frames(data_dir)
        train_frames.append(train)

    for data_dir in test_data_dirs:
        _, test = load_processed_frames(data_dir)
        test_frames.append(test)

    train = pd.concat(train_frames, axis=0, ignore_index=True, sort=False)
    test = pd.concat(test_frames, axis=0, ignore_index=True, sort=False)
    x_train, y_train = split_features_targets(train)
    x_test, y_test = split_features_targets(test)
    x_train, x_test = x_train.align(x_test, join="outer", axis=1, fill_value=0)

    metadata = {
        "feature_names": x_train.columns.tolist(),
        "train_data_dirs": [str(path) for path in train_data_dirs],
        "test_data_dirs": [str(path) for path in test_data_dirs],
    }

    if return_metadata:
        return x_train, x_test, y_train, y_test, metadata

    return x_train, x_test, y_train, y_test


def get_output_dir(data_dirs):
    """Build output directory from processed dataset directory names."""
    data_dirs = resolve_data_dirs(data_dirs)
    dataset_name = "_".join(data_dir.name for data_dir in data_dirs)
    return ML_RESULTS_DIR / dataset_name


def get_train_models_dir(data_dirs):
    """Build train_models output directory."""
    return get_output_dir(data_dirs) / "train_models"


def get_transfer_output_dir(train_data_dirs, test_data_dirs):
    """Build output directory for explicit train -> test experiments."""
    train_name = "_".join(data_dir.name for data_dir in resolve_data_dirs(train_data_dirs))
    test_name = "_".join(data_dir.name for data_dir in resolve_data_dirs(test_data_dirs))
    return ML_RESULTS_DIR / f"{train_name}_to_{test_name}"


def get_transfer_train_models_dir(train_data_dirs, test_data_dirs):
    """Build train_models output directory for explicit transfer experiments."""
    return get_transfer_output_dir(train_data_dirs, test_data_dirs) / "train_models"


def to_numpy(data):
    """Convert pandas objects to numpy arrays before passing them to ML libraries."""
    return data.to_numpy() if hasattr(data, "to_numpy") else data


def get_feature_types(features):
    """Split feature columns into numeric and categorical groups."""
    numeric_features = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        column for column in features.columns if column not in numeric_features
    ]
    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    """Build sklearn preprocessing for numeric and categorical features."""
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ]
    )
    return preprocessor.set_output(transform="pandas")


def build_model_pipeline(model, numeric_features, categorical_features):
    """Wrap a model with shared preprocessing."""
    return Pipeline(
        [
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            ("model", model),
        ]
    )


def build_model_candidates(
    numeric_features,
    categorical_features,
    random_state=TRAIN_MODELS_RANDOM_STATE,
):
    """Create candidate models for a first-pass model class comparison."""

    models = {
        "CatBoost": CatBoostRegressor(
            loss_function="MultiRMSE",
            random_seed=random_state,
            allow_writing_files=False,
            logging_level="Silent",
            thread_count=-1,
        ),
        "XGBoost": MultiOutputRegressor(
            XGBRegressor(
                random_state=random_state,
                n_jobs=-1,
            )
        ),
        "LightGBM": MultiOutputRegressor(
            LGBMRegressor(
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            )
        ),
        "ExtraTrees": ExtraTreesRegressor(
            random_state=random_state,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestRegressor(
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    return {
        model_name: build_model_pipeline(
            model,
            numeric_features,
            categorical_features,
        )
        for model_name, model in models.items()
    }


def evaluate_model(model, x_test, y_test):
    """Calculate multi-output regression metrics for a fitted model."""
    predictions = model.predict(x_test)
    rmse_per_target = np.sqrt(
        mean_squared_error(y_test, predictions, multioutput="raw_values")
    )
    mae_per_target = mean_absolute_error(y_test, predictions, multioutput="raw_values")
    r2_per_target = r2_score(y_test, predictions, multioutput="raw_values")

    return {
        "rmse": float(np.mean(rmse_per_target)),
        "mae": float(np.mean(mae_per_target)),
        "r2": float(np.mean(r2_per_target)),
        "rmse_per_target": dict(zip(y_test.columns, rmse_per_target.tolist())),
        "mae_per_target": dict(zip(y_test.columns, mae_per_target.tolist())),
        "r2_per_target": dict(zip(y_test.columns, r2_per_target.tolist())),
    }


def prefix_metrics(metrics, prefix):
    """Add a prefix to metric names."""
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def cross_validate_model(model, x_train, y_train):
    """Estimate model quality on train data with cross-validation."""
    cv = KFold(n_splits=3, shuffle=True, random_state=TRAIN_MODELS_RANDOM_STATE)
    scores = cross_val_score(
        model,
        x_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
    )
    rmse_scores = -scores
    return {
        "cv_rmse": float(np.mean(rmse_scores)),
        "cv_rmse_std": float(np.std(rmse_scores)),
    }


def train_and_evaluate_models(x_train, x_test, y_train, y_test, candidates=None):
    """Fit all candidates and rank models by train-set cross-validation RMSE."""
    if candidates is None:
        numeric_features, categorical_features = get_feature_types(x_train)
        candidates = build_model_candidates(numeric_features, categorical_features)
    metrics = []

    for model_name, model in candidates.items():
        print(f"Cross-validating {model_name}...")
        cv_metrics = cross_validate_model(model, x_train, y_train)
        print(f"Training {model_name}...")
        model.fit(x_train, y_train)
        train_metrics = evaluate_model(model, x_train, y_train)
        test_metrics = evaluate_model(model, x_test, y_test)
        metrics.append(
            {
                "model": model_name,
                **cv_metrics,
                **test_metrics,
                **prefix_metrics(train_metrics, "train"),
                **prefix_metrics(test_metrics, "test"),
            }
        )

    metrics_df = pd.DataFrame(metrics).sort_values(
        by=["cv_rmse", "cv_rmse_std"],
        ascending=[True, True],
    )
    best_model_name = metrics_df.iloc[0]["model"]

    return best_model_name, metrics_df


def save_training_results(
    metrics_df,
    results_dir,
):
    """Save all model-selection metrics."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "model_selection_metrics.csv"

    metrics_df.to_csv(metrics_path, index=False)

    return metrics_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train models on processed RAG data.")
    parser.add_argument(
        "data_dirs",
        nargs="*",
        type=Path,
        help="Path(s) to directories with processed train.csv and test.csv files.",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        dest="data_dir_options",
        action="append",
        nargs="+",
        type=Path,
        help=(
            "Path(s) to processed train/test data. Can be used multiple times. "
        ),
    )
    parser.add_argument(
        "--train-data-dir",
        dest="train_data_dir_options",
        action="append",
        nargs="+",
        type=Path,
        help="Source processed dataset directory/directories used for training.",
    )
    parser.add_argument(
        "--test-data-dir",
        dest="test_data_dir_options",
        action="append",
        nargs="+",
        type=Path,
        help="Target processed dataset directory/directories used for testing.",
    )

    args = parser.parse_args()
    option_dirs = [
        data_dir
        for data_dir_group in (args.data_dir_options or [])
        for data_dir in data_dir_group
    ]
    args.data_dirs = option_dirs or args.data_dirs
    args.train_data_dirs = [
        data_dir
        for data_dir_group in (args.train_data_dir_options or [])
        for data_dir in data_dir_group
    ]
    args.test_data_dirs = [
        data_dir
        for data_dir_group in (args.test_data_dir_options or [])
        for data_dir in data_dir_group
    ]
    if args.data_dirs and str(args.data_dirs[0]) == "data_dirs":
        args.data_dirs = args.data_dirs[1:]
    del args.data_dir_options
    del args.train_data_dir_options
    del args.test_data_dir_options
    if bool(args.train_data_dirs) != bool(args.test_data_dirs):
        parser.error("--train-data-dir and --test-data-dir must be used together")
    return args


def main():
    args = parse_args()
    if args.train_data_dirs and args.test_data_dirs:
        output_dir = get_transfer_train_models_dir(args.train_data_dirs, args.test_data_dirs)
        x_train, x_test, y_train, y_test = load_transfer_data(
            args.train_data_dirs,
            args.test_data_dirs,
        )
        loaded_data_message = (
            "Loaded train data from "
            f"{', '.join(str(path) for path in args.train_data_dirs)} "
            "and test data from "
            f"{', '.join(str(path) for path in args.test_data_dirs)}"
        )
    else:
        output_dir = get_train_models_dir(args.data_dirs)
        x_train, x_test, y_train, y_test = load_processed_data(args.data_dirs)
        loaded_data_message = (
            "Loaded processed data from "
            f"{', '.join(str(path) for path in args.data_dirs)}"
        )
    best_model_name, metrics_df = train_and_evaluate_models(
        x_train, x_test, y_train, y_test
    )
    metrics_path = save_training_results(
        metrics_df,
        results_dir=output_dir,
    )

    print("\nModel selection results:")
    print(
        metrics_df[
            [
                "model",
                "cv_rmse",
                "cv_rmse_std",
                "train_rmse",
                "train_mae",
                "train_r2",
                "test_rmse",
                "test_mae",
                "test_r2",
            ]
        ].to_string(index=False)
    )
    print(f"\nBest model by CV RMSE: {best_model_name}")
    print(loaded_data_message)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
