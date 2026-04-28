import pickle
from pathlib import Path

import pandas as pd

from ML.constants import (
    PROJECT_ROOT,
    SUMMARY_METADATA_COLUMNS,
    SUMMARY_PATH,
)
from ML.train_models import get_output_dir, load_processed_data


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


def get_interpretability_output_dir(data_dirs, name):
    """Build an interpretability output directory for selected dataset(s)."""
    return get_output_dir(data_dirs) / name


def sanitize_name(name):
    """Make target names safe for filenames."""
    return str(name).replace("/", "_").replace("\\", "_").replace(" ", "_")
