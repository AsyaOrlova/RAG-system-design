import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "oxazo"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RANDOM_STATE = 42

TARGET_COLUMNS = [
    "bert_score_recall",
    "cosine_similarity",
    "golden_doi_mrr",
    "golden_doi_recall",
    "numeric_recall",
    "rouge_l_recall",
    "scientific_fact_recall",
    "source_support_hit",
    "source_support_max_similarity",
    "token_f1",
    "token_recall",
]


def split_features_targets(data, target_columns=None):
    """Split a processed dataframe into feature and target matrices."""
    target_columns = target_columns or TARGET_COLUMNS
    missing_targets = [column for column in target_columns if column not in data.columns]

    if missing_targets:
        raise ValueError(f"Missing target columns in processed data: {missing_targets}")

    return data.drop(columns=target_columns), data[target_columns]


def load_processed_data(data_dir=PROCESSED_DATA_DIR):
    """Load train/test files and split them into X/y for model training."""
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    missing_files = [str(path) for path in [train_path, test_path] if not path.exists()]

    if missing_files:
        raise FileNotFoundError(
            "Processed train/test files were not found. Run src/preprocess.py first. "
            f"Missing: {missing_files}"
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    x_train, y_train = split_features_targets(train)
    x_test, y_test = split_features_targets(test)

    return x_train, x_test, y_train, y_test


def to_numpy(data):
    """Convert pandas objects to numpy arrays before passing them to ML libraries."""
    return data.to_numpy() if hasattr(data, "to_numpy") else data


def build_model_candidates(random_state=RANDOM_STATE):
    """Create candidate models for a first-pass model class comparison."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    return {
        "CatBoost": CatBoostRegressor(
            iterations=400,
            learning_rate=0.04,
            depth=4,
            loss_function="MultiRMSE",
            random_seed=random_state,
            allow_writing_files=False,
            logging_level="Silent",
            thread_count=-1,
        ),
        "XGBoost": MultiOutputRegressor(
            XGBRegressor(
                n_estimators=400,
                learning_rate=0.04,
                max_depth=3,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=random_state,
                n_jobs=-1,
            )
        ),
        "LightGBM": MultiOutputRegressor(
            LGBMRegressor(
                n_estimators=400,
                learning_rate=0.04,
                num_leaves=15,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            )
        ),
        "HistGradientBoosting": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                learning_rate=0.06,
                max_iter=300,
                max_leaf_nodes=15,
                l2_regularization=0.01,
                random_state=random_state,
            )
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def evaluate_model(model, x_test, y_test):
    """Calculate multi-output regression metrics for a fitted model."""
    predictions = model.predict(to_numpy(x_test))
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


def train_and_evaluate_models(x_train, x_test, y_train, y_test, candidates=None):
    """Fit all candidates and select the model with the lowest test RMSE."""
    candidates = candidates or build_model_candidates()
    trained_models = {}
    metrics = []
    x_train_values = to_numpy(x_train)
    y_train_values = to_numpy(y_train)

    for model_name, model in candidates.items():
        print(f"Training {model_name}...")
        model.fit(x_train_values, y_train_values)
        model_metrics = evaluate_model(model, x_test, y_test)
        trained_models[model_name] = model
        metrics.append({"model": model_name, **model_metrics})

    metrics_df = pd.DataFrame(metrics).sort_values(
        by=["rmse", "mae", "r2"],
        ascending=[True, True, False],
    )
    best_model_name = metrics_df.iloc[0]["model"]

    return trained_models[best_model_name], best_model_name, metrics_df


def save_training_results(
    best_model,
    best_model_name,
    metrics_df,
    models_dir=MODELS_DIR,
    results_dir=RESULTS_DIR,
):
    """Save the best model and all model-selection metrics."""
    models_dir = Path(models_dir)
    results_dir = Path(results_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = models_dir / "best_model.pkl"
    metrics_path = results_dir / "model_selection_metrics.csv"
    metadata_path = results_dir / "best_model_metadata.json"

    with best_model_path.open("wb") as model_file:
        pickle.dump(best_model, model_file)

    metrics_df.to_csv(metrics_path, index=False)

    metadata = {
        "best_model": best_model_name,
        "selection_metric": "lowest mean test RMSE",
        "best_model_path": str(best_model_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics_df.iloc[0].to_dict(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return best_model_path, metrics_path, metadata_path


def main():
    x_train, x_test, y_train, y_test = load_processed_data()
    best_model, best_model_name, metrics_df = train_and_evaluate_models(
        x_train, x_test, y_train, y_test
    )
    best_model_path, metrics_path, metadata_path = save_training_results(
        best_model, best_model_name, metrics_df
    )

    print("\nModel selection results:")
    print(metrics_df[["model", "rmse", "mae", "r2"]].to_string(index=False))
    print(f"\nBest model: {best_model_name}")
    print(f"Saved best model to {best_model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
