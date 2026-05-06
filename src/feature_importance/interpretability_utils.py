import pickle
from pathlib import Path

import pandas as pd

from ml.constants import (
    PROJECT_ROOT,
    SUMMARY_METADATA_COLUMNS,
    SUMMARY_PATH,
)
from ml.train_models import get_output_dir, load_processed_data


EXCLUDED_FEATURES = {"chunk_overlap", "token_overlap"}


def log(message):
    """Print progress immediately for long-running interpretability calculations."""
    print(message, flush=True)


def infer_data_dirs_from_model_path(model_path):
    """Infer processed dataset directories from a saved model path."""
    model_path = Path(model_path)
    experiment_name = model_path.parent.parent.name
    candidate_dirs = []

    for name in experiment_name.split("_"):
        candidate_dir = PROJECT_ROOT / "data" / "processed" / name
        if candidate_dir.exists():
            candidate_dirs.append(candidate_dir)

    return candidate_dirs


def load_model(model_path):
    """Load a saved model bundle or raw estimator from disk."""
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            "Saved model was not found. Run training or tuning first. "
            f"Missing: {model_path}"
        )

    with model_path.open("rb") as model_file:
        saved_object = pickle.load(model_file)

    if isinstance(saved_object, dict) and "model" in saved_object:
        return saved_object["model"], model_path

    return saved_object, model_path


def load_features(data_dirs):
    """Load processed training features from one or more datasets."""
    x_train, _, _, _ = load_processed_data(data_dirs)
    return x_train


def get_summary_feature_columns(summary_path=SUMMARY_PATH):
    """Return feature columns from summary.csv, excluding metadata columns."""
    summary_path = Path(summary_path)
    if not summary_path.exists():
        return []

    summary = pd.read_csv(summary_path, nrows=1)
    return [
        column
        for column in summary.columns
        if column not in SUMMARY_METADATA_COLUMNS
    ]


def get_excluded_feature_names(features):
    """Return summary and manually excluded feature names present in features."""
    excluded = set(get_summary_feature_columns()) | EXCLUDED_FEATURES
    return [column for column in features.columns if column in excluded]


def get_analysis_feature_names(features):
    """Return feature names included in interpretability outputs."""
    excluded = set(get_excluded_feature_names(features))
    return [column for column in features.columns if column not in excluded]


def unwrap_pipeline(model, features):
    """Apply a fitted preprocessing pipeline and return the final estimator."""
    if not hasattr(model, "named_steps"):
        return model, features

    if "preprocess" not in model.named_steps or "model" not in model.named_steps:
        return model, features

    preprocessor = model.named_steps["preprocess"]
    estimator = model.named_steps["model"]
    transformed = preprocessor.transform(features)

    if isinstance(transformed, pd.DataFrame):
        transformed_features = transformed
    else:
        transformed_features = pd.DataFrame(
            transformed,
            columns=preprocessor.get_feature_names_out(),
            index=features.index,
        )

    return estimator, transformed_features


def get_original_feature_name(transformed_feature_name, original_feature_names):
    """Map a ColumnTransformer output name back to its source feature name."""
    feature_name = transformed_feature_name.split("__", 1)[-1]

    if feature_name in original_feature_names:
        return feature_name

    for original_name in sorted(original_feature_names, key=len, reverse=True):
        if feature_name.startswith(f"{original_name}_"):
            return original_name

    return feature_name


def get_transformed_analysis_feature_names(
    transformed_features,
    original_features,
    excluded_original_features,
):
    """Return transformed feature names included in interpretability outputs."""
    original_feature_names = set(original_features.columns)
    excluded = set(excluded_original_features)

    return [
        transformed_feature
        for transformed_feature in transformed_features.columns
        if get_original_feature_name(transformed_feature, original_feature_names)
        not in excluded
    ]


def clean_transformed_feature_names(features):
    """Remove ColumnTransformer prefixes from feature names for plots/tables."""
    renamed = features.copy()
    renamed.columns = [column.split("__", 1)[-1] for column in renamed.columns]
    return renamed


def get_interpretability_output_dir(data_dirs, name):
    """Build an interpretability output directory for selected dataset(s)."""
    return get_output_dir(data_dirs) / name


def sanitize_name(name):
    """Make target names safe for filenames."""
    return str(name).replace("/", "_").replace("\\", "_").replace(" ", "_")
