from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "oxazo_results.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "oxazo"

DROP_COLUMNS = [
    "corpus",
    "kb_alias",
    "eval_alias",
    "count_tokens",
    "paragraph_filtrator",
    "table_grabber",
]

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


def load_data(input_path=DEFAULT_INPUT_PATH):
    """Load source data from csv."""
    return pd.read_csv(input_path)


def clean_data(data):
    """Drop non-informative columns and convert bool-like values."""
    cleaned = data.drop(columns=DROP_COLUMNS, errors="ignore").copy()

    for column in cleaned.select_dtypes(include=["bool"]).columns:
        cleaned[column] = cleaned[column].astype(int)

    return cleaned


def split_features_targets(data, target_columns=None):
    """Separate feature matrix and multi-output target matrix."""
    target_columns = target_columns or TARGET_COLUMNS
    missing_targets = [column for column in target_columns if column not in data.columns]

    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    prepared = data.copy()
    prepared[target_columns] = prepared[target_columns].apply(pd.to_numeric, errors="coerce")
    prepared = prepared.dropna(subset=target_columns)

    features = prepared.drop(columns=target_columns)
    targets = prepared[target_columns]

    return features, targets


def fill_missing_values(train_features, test_features):
    """Fill missing values using statistics learned from train data only."""
    train_features = train_features.copy()
    test_features = test_features.copy()

    numeric_columns = train_features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in train_features.columns if column not in numeric_columns
    ]

    for column in numeric_columns:
        fill_value = train_features[column].median()
        train_features[column] = train_features[column].fillna(fill_value)
        test_features[column] = test_features[column].fillna(fill_value)

    for column in categorical_columns:
        mode = train_features[column].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "unknown"
        train_features[column] = train_features[column].fillna(fill_value).astype(str)
        test_features[column] = test_features[column].fillna(fill_value).astype(str)

    return train_features, test_features, numeric_columns, categorical_columns


def encode_categorical_features(train_features, test_features, categorical_columns):
    """One-hot encode categorical columns and align train/test columns."""
    if not categorical_columns:
        return train_features, test_features

    train_encoded = pd.get_dummies(
        train_features, columns=categorical_columns, drop_first=False, dtype=int
    )
    test_encoded = pd.get_dummies(
        test_features, columns=categorical_columns, drop_first=False, dtype=int
    )

    train_encoded, test_encoded = train_encoded.align(
        test_encoded, join="left", axis=1, fill_value=0
    )

    return train_encoded, test_encoded


def standardize_numeric_features(train_features, test_features, numeric_columns):
    """Standardize numeric features with mean and std from train data."""
    if not numeric_columns:
        return train_features, test_features

    train_features = train_features.copy()
    test_features = test_features.copy()
    means = train_features[numeric_columns].mean()
    stds = train_features[numeric_columns].std(ddof=0).replace(0, 1)

    train_features[numeric_columns] = (train_features[numeric_columns] - means) / stds
    test_features[numeric_columns] = (test_features[numeric_columns] - means) / stds

    return train_features, test_features


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


def preprocess_data(data, test_size=0.2, random_state=42):
    """Run the full preprocessing pipeline and return split datasets."""
    cleaned = clean_data(data)
    features, targets = split_features_targets(cleaned)

    x_train, x_test, y_train, y_test = train_test_split_data(
        features=features,
        targets=targets,
        test_size=test_size,
        random_state=random_state,
    )

    x_train, x_test, numeric_columns, categorical_columns = fill_missing_values(
        x_train, x_test
    )
    x_train, x_test = encode_categorical_features(
        x_train, x_test, categorical_columns
    )
    x_train, x_test = standardize_numeric_features(x_train, x_test, numeric_columns)

    return x_train, x_test, y_train, y_test


def save_processed_data(x_train, x_test, y_train, y_test, output_dir=DEFAULT_OUTPUT_DIR):
    """Save processed train/test files with features and targets together."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = pd.concat(
        [x_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
    )
    test = pd.concat(
        [x_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
    )

    train.to_csv(output_dir / "train.csv", index=False)
    test.to_csv(output_dir / "test.csv", index=False)


def main():
    data = load_data()
    x_train, x_test, y_train, y_test = preprocess_data(data)
    save_processed_data(x_train, x_test, y_train, y_test)
    print(f"Processed data saved to {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
