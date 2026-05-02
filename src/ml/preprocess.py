import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml.constants import (
    DROP_COLUMNS,
    PREPROCESS_RESULTS_DIR,
    SUMMARY_METADATA_COLUMNS,
    SUMMARY_PATH,
    TARGET_COLUMNS,
)


def load_data(input_path):
    """Load source data from csv."""
    return pd.read_csv(input_path)


def get_dataset_name(input_path):
    """Extract dataset name from source csv filename."""
    dataset_name = Path(input_path).stem
    if dataset_name.endswith("_results"):
        dataset_name = dataset_name[: -len("_results")]

    return dataset_name


def clean_data(data):
    """Drop non-informative columns and convert bool-like values."""
    cleaned = data.drop(columns=DROP_COLUMNS, errors="ignore").copy()

    for column in cleaned.select_dtypes(include=["bool"]).columns:
        cleaned[column] = cleaned[column].astype(int)

    return cleaned


def load_summary_features(dataset_name, summary_path=SUMMARY_PATH):
    """Load dataset-level summary feature values for one dataset."""
    summary_path = Path(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Dataset summary file was not found: {summary_path}")

    summary = pd.read_csv(summary_path)
    if "dataset" not in summary.columns:
        raise ValueError(f"{summary_path} must contain a 'dataset' column")

    matched_rows = summary[summary["dataset"] == dataset_name]
    if matched_rows.empty:
        raise ValueError(
            f"Dataset '{dataset_name}' was not found in summary file: {summary_path}"
        )

    return matched_rows.iloc[0].drop(
        labels=list(SUMMARY_METADATA_COLUMNS),
        errors="ignore",
    ).to_dict()


def add_summary_features(data, summary_features):
    """Add dataset-level summary metrics as feature columns."""
    enriched = data.copy()

    for column, value in summary_features.items():
        enriched[column] = value

    return enriched


def split_features_targets(data):
    """Separate feature matrix and multi-output target matrix."""
    missing_targets = [column for column in TARGET_COLUMNS if column not in data.columns]

    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    prepared = data.copy()
    prepared[TARGET_COLUMNS] = prepared[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")

    features = prepared.drop(columns=TARGET_COLUMNS)
    targets = prepared[TARGET_COLUMNS]

    return features, targets


def train_test_split_data(features, targets, test_size=0.2, random_state=42):
    """Split features and targets into train/test subsets."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    shuffled_index = features.sample(frac=1, random_state=random_state).index
    test_count = max(1, round(len(shuffled_index) * test_size))
    test_index = shuffled_index[:test_count]
    train_index = shuffled_index[test_count:]

    return (
        features.loc[train_index].copy(),
        features.loc[test_index].copy(),
        targets.loc[train_index].copy(),
        targets.loc[test_index].copy(),
    )


def preprocess_data(
    data,
    dataset_name=None,
    test_size=0.2,
    random_state=42,
    include_summary_features=True,
):
    """Run the full preprocessing pipeline and return split datasets."""
    cleaned = clean_data(data)
    if include_summary_features and dataset_name:
        summary_features = load_summary_features(dataset_name)
        cleaned = add_summary_features(cleaned, summary_features)

    features, targets = split_features_targets(cleaned)

    x_train, x_test, y_train, y_test = train_test_split_data(
        features=features,
        targets=targets,
        test_size=test_size,
        random_state=random_state,
    )

    return x_train, x_test, y_train, y_test


def save_processed_data(x_train, x_test, y_train, y_test, output_dir=None):
    """Save processed train/test files with features and targets together."""
    output_dir = Path(output_dir or PREPROCESS_RESULTS_DIR / "oxazo")
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.concat(
        [x_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
    )
    test = pd.concat(
        [x_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
    )

    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)


def get_output_dir(input_path, summary_features):
    """Build output directory from input dataset name."""
    if summary_features:
        return PREPROCESS_RESULTS_DIR / get_dataset_name(input_path)
    else:
        return PREPROCESS_RESULTS_DIR / f"{get_dataset_name(input_path)}_no_summary"


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess RAG metrics dataset.")
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Path to the input dataset CSV.",
    )
    parser.add_argument(
        "-i",
        "--input-path",
        dest="input_path_option",
        type=Path,
        help=f"Path to the input dataset CSV.",
    )
    parser.add_argument(
        "--summary-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Add dataset-level summary features from summary.csv. "
            "Use --no-summary-features to skip them."
        ),
    )
    args = parser.parse_args()
    args.input_path = args.input_path_option or args.input_path
    if args.input_path is None:
        parser.error("input_path is required")

    del args.input_path_option
    return args


def main():
    args = parse_args()
    data = load_data(args.input_path)
    dataset_name = get_dataset_name(args.input_path)
    x_train, x_test, y_train, y_test = preprocess_data(
        data,
        dataset_name=dataset_name,
        include_summary_features=args.summary_features,
    )
    output_dir = get_output_dir(args.input_path, args.summary_features)
    save_processed_data(x_train, x_test, y_train, y_test, output_dir)
    print(f"Processed {args.input_path} and saved data to {output_dir}")


if __name__ == "__main__":
    main()
