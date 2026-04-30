import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


DEFAULT_TARGET_COLUMNS = [
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

SUMMARY_METADATA_COLUMNS = {"dataset", "articles_file", "chunks_file", "num_documents"}


def load_pickle_model(model_path):
    """Load a surrogate model saved either directly or as a model bundle."""
    model_path = Path(model_path)
    with model_path.open("rb") as model_file:
        saved_object = pickle.load(model_file)

    if isinstance(saved_object, dict) and "model" in saved_object:
        model = saved_object["model"]
        set_unknown_categories_to_ignore(model)
        return model, saved_object.get("preprocessing", {})

    set_unknown_categories_to_ignore(saved_object)
    return saved_object, {}


def set_unknown_categories_to_ignore(model):
    """Make saved sklearn pipelines robust to unseen categorical values."""
    named_steps = getattr(model, "named_steps", {})
    preprocessor = named_steps.get("preprocess") if named_steps else None
    if preprocessor is None:
        return

    transformers = []
    if hasattr(preprocessor, "transformers"):
        transformers.extend(preprocessor.transformers)
    if hasattr(preprocessor, "transformers_"):
        transformers.extend(preprocessor.transformers_)

    for _, transformer, _ in transformers:
        set_encoder_unknown_categories_to_ignore(transformer)

    named_transformers = getattr(preprocessor, "named_transformers_", {})
    for transformer in getattr(named_transformers, "values", lambda: [])():
        set_encoder_unknown_categories_to_ignore(transformer)


def set_encoder_unknown_categories_to_ignore(transformer):
    """Update OneHotEncoder instances, including encoders inside pipelines."""
    if isinstance(transformer, OneHotEncoder):
        transformer.handle_unknown = "ignore"
        return

    named_steps = getattr(transformer, "named_steps", {})
    for step in getattr(named_steps, "values", lambda: [])():
        set_encoder_unknown_categories_to_ignore(step)


def get_expected_feature_names(model, metadata=None):
    """Infer feature names expected by the fitted surrogate model."""
    metadata = metadata or {}
    if metadata.get("feature_names"):
        return list(metadata["feature_names"])

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    named_steps = getattr(model, "named_steps", {})
    preprocessor = named_steps.get("preprocess")
    if preprocessor is not None and hasattr(preprocessor, "feature_names_in_"):
        return list(preprocessor.feature_names_in_)

    return None


def load_summary_features(dataset, summary_path):
    """Load dataset-level features used during model training."""
    if not dataset:
        return {}

    summary_path = Path(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Dataset summary file was not found: {summary_path}")

    summary = pd.read_csv(summary_path)
    matched_rows = summary[summary["dataset"] == dataset]
    if matched_rows.empty:
        raise ValueError(f"Dataset '{dataset}' was not found in {summary_path}")

    return matched_rows.iloc[0].drop(
        labels=list(SUMMARY_METADATA_COLUMNS),
        errors="ignore",
    ).to_dict()


def make_feature_frame(params, expected_feature_names=None, extra_features=None):
    """Create a one-row dataframe and align it to model feature names if available."""
    row = dict(params)
    row.update(extra_features or {})
    features = pd.DataFrame([row])

    if expected_feature_names is None:
        return features

    for column in expected_feature_names:
        if column not in features.columns:
            features[column] = 0

    return features[expected_feature_names]


class SurrogatePredictor:
    """Predict RAG metrics from parameter dictionaries using a saved surrogate."""

    def __init__(
        self,
        model,
        metadata=None,
        target_columns=None,
        extra_features=None,
    ):
        self.model = model
        self.metadata = metadata or {}
        self.target_columns = list(target_columns or DEFAULT_TARGET_COLUMNS)
        self.extra_features = extra_features or {}
        self.expected_feature_names = get_expected_feature_names(
            self.model,
            self.metadata,
        )

    @classmethod
    def from_pickle(
        cls,
        model_path,
        target_columns=None,
        extra_features=None,
    ):
        model, metadata = load_pickle_model(model_path)
        return cls(
            model=model,
            metadata=metadata,
            target_columns=target_columns,
            extra_features=extra_features,
        )

    def predict(self, params):
        features = make_feature_frame(
            params=params,
            expected_feature_names=self.expected_feature_names,
            extra_features=self.extra_features,
        )
        prediction = self.model.predict(features)
        values = prediction[0] if getattr(prediction, "ndim", 1) > 1 else prediction
        values = np.atleast_1d(values)

        if len(values) != len(self.target_columns):
            raise ValueError(
                "The surrogate returned "
                f"{len(values)} values, but {len(self.target_columns)} target names "
                "were provided."
            )

        return dict(zip(self.target_columns, [float(value) for value in values]))
