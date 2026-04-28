import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ML.constants import TARGET_COLUMNS
from ml.interpretability_utils import (
    get_analysis_feature_names,
    get_excluded_feature_names,
    get_interpretability_output_dir,
    infer_data_dirs_from_model_path,
    load_features,
    load_model,
    log,
    sanitize_name,
)
from ML.train_models import resolve_data_dirs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate SHAP feature importances for each target and their mean."
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=Path,
        required=True,
        help="Path to a saved model.",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        dest="data_dir_options",
        action="append",
        nargs="+",
        type=Path,
        help="Path(s) to processed train/test data used to compute SHAP values.",
    )

    args = parser.parse_args()
    option_dirs = [
        data_dir
        for data_dir_group in (args.data_dir_options or [])
        for data_dir in data_dir_group
    ]
    args.data_dirs = option_dirs or infer_data_dirs_from_model_path(args.model_path)
    del args.data_dir_options
    return args


def get_shap_output_dir(data_dirs):
    return get_interpretability_output_dir(data_dirs, "shap_feature_importance")


def get_output_paths(output_dir, suffix):
    output_dir = Path(output_dir)
    return (
        output_dir / f"shap_values_{suffix}.csv",
        output_dir / f"shap_values_{suffix}_beeswarm.png",
    )


def get_dataset_label(data_dirs):
    """Build a readable dataset label for plot titles."""
    return "_".join(data_dir.name for data_dir in resolve_data_dirs(data_dirs))


def normalize_direct_shap_values(raw_values, target_names, n_rows, n_features):
    """Convert direct multi-output SHAP output into target -> matrix mapping."""
    if isinstance(raw_values, list):
        if len(raw_values) != len(target_names):
            raise ValueError(
                "SHAP output target count does not match TARGET_COLUMNS: "
                f"{len(raw_values)} vs {len(target_names)}."
            )
        return {
            target_name: np.asarray(values)
            for target_name, values in zip(target_names, raw_values)
        }

    values = np.asarray(raw_values)
    if values.ndim == 2:
        if len(target_names) != 1:
            raise ValueError(
                "Received 2D SHAP values for a multi-target model. "
                "Use a supported multi-output tree model or MultiOutputRegressor."
            )
        return {target_names[0]: values}

    if values.ndim != 3:
        raise ValueError(f"Unsupported SHAP values shape: {values.shape}")

    if values.shape == (n_rows, n_features, len(target_names)):
        return {
            target_name: values[:, :, target_index]
            for target_index, target_name in enumerate(target_names)
        }

    if values.shape == (len(target_names), n_rows, n_features):
        return {
            target_name: values[target_index, :, :]
            for target_index, target_name in enumerate(target_names)
        }

    raise ValueError(f"Unsupported multi-output SHAP values shape: {values.shape}")


def compute_shap_values_by_target(model, features, target_names):
    """Compute SHAP values for each target separately."""
    if hasattr(model, "estimators_"):
        if len(model.estimators_) != len(target_names):
            raise ValueError(
                "Estimator count does not match target count: "
                f"{len(model.estimators_)} estimators for {len(target_names)} targets."
            )

        shap_values_by_target = {}
        for target_name, estimator in zip(target_names, model.estimators_):
            log(f"  Computing SHAP values for target: {target_name}")
            explainer = shap.TreeExplainer(estimator)
            shap_values_by_target[target_name] = np.asarray(
                explainer.shap_values(features)
            )
        return shap_values_by_target

    log("  Computing SHAP values for direct multi-output model...")
    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(features)
    return normalize_direct_shap_values(
        raw_values,
        target_names,
        n_rows=len(features),
        n_features=len(features.columns),
    )


def filter_shap_values(shap_values, features, analysis_feature_names):
    """Keep SHAP columns selected for reporting."""
    positions = [features.columns.get_loc(column) for column in analysis_feature_names]
    return np.asarray(shap_values)[:, positions]


def save_shap_outputs(shap_values, features, csv_path, plot_path, title):
    """Save mean SHAP importances and a beeswarm plot."""
    csv_path = Path(csv_path)
    plot_path = Path(plot_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    importance_df = pd.DataFrame(
        {
            "feature": features.columns,
            "mean_shap": np.mean(shap_values, axis=0),
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, max(6, 0.35 * min(20, len(features.columns)))))
    shap.summary_plot(
        shap_values,
        features,
        feature_names=features.columns.tolist(),
        max_display=20,
        show=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    log("Loading model...")
    model, model_path = load_model(args.model_path)
    log(f"Loaded model from {model_path}")

    log("Loading processed features...")
    features = load_features(args.data_dirs)
    excluded_features = get_excluded_feature_names(features)
    analysis_feature_names = get_analysis_feature_names(features)
    analysis_features = features[analysis_feature_names]
    log(
        f"Loaded {len(features)} rows and {len(features.columns)} model features. "
        f"Reporting {len(analysis_feature_names)} features after exclusions."
    )
    if excluded_features:
        log(f"Excluded from SHAP outputs: {', '.join(excluded_features)}")

    output_dir = get_shap_output_dir(args.data_dirs)
    dataset_label = get_dataset_label(args.data_dirs)
    log(f"SHAP outputs will be saved to {output_dir}")

    log("Computing SHAP values...")
    shap_values_by_target = compute_shap_values_by_target(
        model,
        features,
        TARGET_COLUMNS,
    )

    filtered_values_by_target = {}
    for target_name, target_values in shap_values_by_target.items():
        filtered_values = filter_shap_values(
            target_values,
            features,
            analysis_feature_names,
        )
        filtered_values_by_target[target_name] = filtered_values

        log(f"Saving SHAP outputs for target: {target_name}")
        target_suffix = sanitize_name(target_name)
        csv_path, plot_path = get_output_paths(output_dir, target_suffix)
        save_shap_outputs(
            filtered_values,
            analysis_features,
            csv_path,
            plot_path,
            f"SHAP Feature Importance ({dataset_label}, {target_name})",
        )

    log("Averaging SHAP values across targets...")
    mean_shap_values = np.mean(
        np.stack(list(filtered_values_by_target.values()), axis=0),
        axis=0,
    )
    all_csv_path, all_plot_path = get_output_paths(output_dir, "all_metrics")
    save_shap_outputs(
        mean_shap_values,
        analysis_features,
        all_csv_path,
        all_plot_path,
        f"SHAP Feature Importance ({dataset_label}, mean across all targets)",
    )

    log("SHAP feature importance calculation finished.")
    log(f"Saved all-target SHAP table to {all_csv_path}")
    log(f"Saved all-target beeswarm plot to {all_plot_path}")


if __name__ == "__main__":
    main()
