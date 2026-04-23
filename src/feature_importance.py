import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from train_models import (
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    split_features_targets,
)


DEFAULT_MODEL_PATH = MODELS_DIR / "best_tuned_model.pkl"
FALLBACK_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SHAP_VALUES_PATH = RESULTS_DIR / "shap_values.npy"
IMPORTANCE_CSV_PATH = RESULTS_DIR / "tree_shap_feature_importance.csv"
IMPORTANCE_PLOT_PATH = RESULTS_DIR / "tree_shap_feature_importance.png"


def load_model(model_path=None):
    """Load tuned model if it exists, otherwise load baseline best model."""
    model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    if not model_path.exists() and model_path == DEFAULT_MODEL_PATH:
        model_path = FALLBACK_MODEL_PATH

    if not model_path.exists():
        raise FileNotFoundError(
            "Saved model was not found. Run training or tuning first. "
            f"Missing: {model_path}"
        )

    with model_path.open("rb") as model_file:
        return pickle.load(model_file), model_path


def load_features(data_dir=PROCESSED_DATA_DIR):
    """Load processed features from train.csv."""
    train_path = Path(data_dir) / "train.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            "Processed train.csv was not found. Run src/preprocess.py first. "
            f"Missing: {train_path}"
        )

    train = pd.read_csv(train_path)
    features, _ = split_features_targets(train)

    return features


def normalize_shap_values(shap_values):
    """Convert TreeExplainer output to an array shaped as rows x features."""
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=0).mean(axis=0)

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = np.mean(shap_values, axis=-1)

    if shap_values.ndim != 2:
        raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")

    return shap_values


def calculate_tree_shap_values(model, features):
    """Calculate TreeSHAP values for direct or multi-output tree models."""
    if hasattr(model, "estimators_"):
        shap_values_by_target = []

        for estimator in model.estimators_:
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(features)
            shap_values_by_target.append(normalize_shap_values(shap_values))

        return np.mean(np.stack(shap_values_by_target, axis=0), axis=0)

    explainer = shap.TreeExplainer(model)
    return normalize_shap_values(explainer.shap_values(features))


def calculate_feature_importance(shap_values, feature_names):
    """Calculate mean absolute TreeSHAP importance by feature."""
    importance_values = np.abs(shap_values).mean(axis=0)

    if len(importance_values) != len(feature_names):
        raise ValueError(
            "SHAP importance length does not match feature count: "
            f"{len(importance_values)} importances for {len(feature_names)} features."
        )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": importance_values,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    total_importance = importance_df["mean_abs_shap"].sum()
    if total_importance > 0:
        importance_df["mean_abs_shap_normalized"] = (
            importance_df["mean_abs_shap"] / total_importance
        )
    else:
        importance_df["mean_abs_shap_normalized"] = 0.0

    return importance_df


def save_outputs(shap_values, importance_df):
    """Save SHAP values, importance table, and plot."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SHAP_VALUES_PATH, shap_values)
    importance_df.to_csv(IMPORTANCE_CSV_PATH, index=False)


def plot_feature_importance(importance_df, output_path=IMPORTANCE_PLOT_PATH, top_n=20):
    """Build and save a horizontal TreeSHAP importance plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top_features = importance_df.head(top_n).sort_values("mean_abs_shap")

    plt.figure(figsize=(10, max(5, 0.35 * len(top_features))))
    plt.barh(top_features["feature"], top_features["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title(f"Top {len(top_features)} Features by TreeSHAP Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def main():
    model, model_path = load_model()
    features = load_features()
    shap_values = calculate_tree_shap_values(model, features)
    importance_df = calculate_feature_importance(shap_values, features.columns)
    save_outputs(shap_values, importance_df)
    plot_path = plot_feature_importance(importance_df)

    print(f"Loaded model from {model_path}")
    print("\nTop TreeSHAP feature importances:")
    print(importance_df.head(20).to_string(index=False))
    print(f"\nSaved SHAP values to {SHAP_VALUES_PATH}")
    print(f"Saved TreeSHAP importance table to {IMPORTANCE_CSV_PATH}")
    print(f"Saved TreeSHAP importance plot to {plot_path}")


if __name__ == "__main__":
    main()
