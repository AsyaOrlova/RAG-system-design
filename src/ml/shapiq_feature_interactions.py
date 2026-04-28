import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapiq.explainer.tree import TreeExplainer

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate shapiq feature interactions for each target and their mean."
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
        help="Path(s) to processed train/test data used to compute interactions.",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=30,
        help="Maximum number of strongest feature interactions to draw.",
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


def get_interactions_output_dir(data_dirs):
    return get_interpretability_output_dir(data_dirs, "shapiq_feature_interactions")


def get_output_paths(output_dir, suffix):
    output_dir = Path(output_dir)
    return (
        output_dir / f"feature_interactions_{suffix}.csv",
        output_dir / f"feature_interactions_{suffix}.json",
        output_dir / f"feature_interactions_{suffix}_graph.png",
    )


def mean_interaction_values(interaction_values_list):
    """Aggregate InteractionValues objects with arithmetic mean."""
    if not interaction_values_list:
        raise ValueError("Expected at least one InteractionValues object to aggregate.")

    aggregated = interaction_values_list[0]
    if len(interaction_values_list) == 1:
        return aggregated

    return aggregated.aggregate(interaction_values_list[1:], aggregation="mean")


def explain_estimator_rows(estimator, features, *, max_order, index):
    """Explain each row of a single-output tree estimator with shapiq."""
    log(
        f"    Explaining {len(features)} rows with "
        f"{estimator.__class__.__name__} (index={index}, max_order={max_order})..."
    )
    explainer = TreeExplainer(
        estimator,
        max_order=max_order,
        min_order=1,
        index=index,
    )

    feature_array = features.to_numpy(dtype=float)
    return [explainer.explain_function(row) for row in feature_array]


def explain_direct_model_target_rows(model, features, *, max_order, index, target_index):
    """Explain one target of a direct multi-output tree model with shapiq."""
    log(
        f"    Explaining {len(features)} rows for target #{target_index} with "
        f"{model.__class__.__name__} (index={index}, max_order={max_order})..."
    )
    explainer = TreeExplainer(
        model,
        max_order=max_order,
        min_order=1,
        index=index,
        class_index=target_index,
    )

    feature_array = features.to_numpy(dtype=float)
    return [explainer.explain_function(row) for row in feature_array]


def compute_interactions_by_target(model, features, target_names):
    """Compute second-order shapiq explanations for each target separately."""
    if hasattr(model, "estimators_"):
        if len(model.estimators_) != len(target_names):
            raise ValueError(
                "Estimator count does not match target count: "
                f"{len(model.estimators_)} estimators for {len(target_names)} targets."
            )

        explanations_by_target = {}
        for target_name, estimator in zip(target_names, model.estimators_):
            log(f"  Computing shapiq interactions for target: {target_name}")
            explanations_by_target[target_name] = explain_estimator_rows(
                estimator,
                features,
                max_order=2,
                index="k-SII",
            )
        return explanations_by_target

    if len(target_names) == 1:
        log(f"  Computing shapiq interactions for target: {target_names[0]}")
        return {
            target_names[0]: explain_estimator_rows(
                model,
                features,
                max_order=2,
                index="k-SII",
            )
        }

    explanations_by_target = {}
    for target_index, target_name in enumerate(target_names):
        log(f"  Computing shapiq interactions for target: {target_name}")
        explanations_by_target[target_name] = explain_direct_model_target_rows(
            model,
            features,
            max_order=2,
            index="k-SII",
            target_index=target_index,
        )
    return explanations_by_target


def average_explanations_across_targets(explanations_by_target):
    """Average explanations across targets for each row."""
    target_names = list(explanations_by_target.keys())
    first_target = target_names[0]
    n_rows = len(explanations_by_target[first_target])

    for target_name in target_names[1:]:
        if len(explanations_by_target[target_name]) != n_rows:
            raise ValueError("Targets contain a different number of row explanations.")

    averaged = []
    for row_index in range(n_rows):
        row_explanations = [
            explanations_by_target[target_name][row_index]
            for target_name in target_names
        ]
        averaged.append(mean_interaction_values(row_explanations))

    return averaged


def pair_interactions_to_frame(interaction_values, feature_names, analysis_feature_names):
    """Convert second-order interaction values into a filtered dataframe."""
    analysis_features = set(analysis_feature_names)
    rows = []

    if hasattr(interaction_values, "interaction_lookup") and hasattr(
        interaction_values, "values"
    ):
        for interaction, value_index in interaction_values.interaction_lookup.items():
            if len(interaction) != 2:
                continue
            first_index, second_index = interaction
            first_feature = feature_names[first_index]
            second_feature = feature_names[second_index]
            if (
                first_feature not in analysis_features
                or second_feature not in analysis_features
            ):
                continue
            value = float(interaction_values.values[value_index])
            rows.append(
                {
                    "feature_1": first_feature,
                    "feature_2": second_feature,
                    "interaction": value,
                    "abs_interaction": abs(value),
                }
            )
    else:
        second_order_values = np.asarray(
            interaction_values.get_n_order_values(2),
            dtype=float,
        )
        if second_order_values.ndim != 2:
            raise ValueError(
                "Expected second-order interaction values to be a matrix, "
                f"got shape {second_order_values.shape}."
            )
        for first_index in range(second_order_values.shape[0]):
            for second_index in range(first_index + 1, second_order_values.shape[1]):
                first_feature = feature_names[first_index]
                second_feature = feature_names[second_index]
                if (
                    first_feature not in analysis_features
                    or second_feature not in analysis_features
                ):
                    continue
                value = float(second_order_values[first_index, second_index])
                rows.append(
                    {
                        "feature_1": first_feature,
                        "feature_2": second_feature,
                        "interaction": value,
                        "abs_interaction": abs(value),
                    }
                )

    columns = ["feature_1", "feature_2", "interaction", "abs_interaction"]
    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, columns=columns).sort_values(
        "abs_interaction",
        ascending=False,
    )


def save_interaction_json(interactions_df, json_path):
    """Save filtered interaction values to JSON."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    records = interactions_df.to_dict(orient="records")
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(records, output_file, indent=2)


def draw_interaction_graph(interactions_df, plot_path, title, max_edges=30):
    """Draw a simple graph for the strongest filtered feature interactions."""
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    strongest = interactions_df.head(max_edges).copy()

    plt.figure(figsize=(12, 12))
    if strongest.empty:
        plt.text(0.5, 0.5, "No interactions to display", ha="center", va="center")
        plt.axis("off")
        plt.title(title)
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        return

    nodes = sorted(set(strongest["feature_1"]) | set(strongest["feature_2"]))
    angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
    positions = {
        node: np.array([np.cos(angle), np.sin(angle)])
        for node, angle in zip(nodes, angles)
    }
    node_strength = {node: 0.0 for node in nodes}
    for _, row in strongest.iterrows():
        node_strength[row["feature_1"]] += row["abs_interaction"]
        node_strength[row["feature_2"]] += row["abs_interaction"]

    max_abs = strongest["abs_interaction"].max() or 1
    for _, row in strongest.iterrows():
        start = positions[row["feature_1"]]
        end = positions[row["feature_2"]]
        color = "#2f6fbb" if row["interaction"] >= 0 else "#c44536"
        width = 0.75 + 5 * row["abs_interaction"] / max_abs
        plt.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=width,
            alpha=0.55,
        )

    max_node_strength = max(node_strength.values()) or 1
    for node, position in positions.items():
        size = 250 + 1250 * node_strength[node] / max_node_strength
        plt.scatter(position[0], position[1], s=size, color="#f2f2f2", edgecolor="#222")
        label_position = position * 1.16
        plt.text(
            label_position[0],
            label_position[1],
            node,
            ha="center",
            va="center",
            fontsize=9,
        )

    plt.title(title)
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_interaction_outputs(
    interaction_values,
    feature_names,
    analysis_feature_names,
    csv_path,
    json_path,
    plot_path,
    title,
    max_edges,
):
    """Save filtered pair interactions as CSV, JSON, and a graph."""
    interactions_df = pair_interactions_to_frame(
        interaction_values,
        feature_names,
        analysis_feature_names,
    )
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    interactions_df.to_csv(csv_path, index=False)
    save_interaction_json(interactions_df, json_path)
    draw_interaction_graph(interactions_df, plot_path, title, max_edges=max_edges)


def main():
    args = parse_args()

    log("Loading model...")
    model, model_path = load_model(args.model_path)
    log(f"Loaded model from {model_path}")

    log("Loading processed features...")
    features = load_features(args.data_dirs)
    excluded_features = get_excluded_feature_names(features)
    analysis_feature_names = get_analysis_feature_names(features)
    log(
        f"Loaded {len(features)} rows and {len(features.columns)} model features. "
        f"Reporting interactions among {len(analysis_feature_names)} features."
    )
    if excluded_features:
        log(f"Excluded from interaction outputs: {', '.join(excluded_features)}")

    output_dir = get_interactions_output_dir(args.data_dirs)
    log(f"Interaction outputs will be saved to {output_dir}")

    log("Computing second-order shapiq interaction values (k-SII)...")
    interactions_by_target = compute_interactions_by_target(
        model,
        features,
        TARGET_COLUMNS,
    )

    for target_name, target_interactions in interactions_by_target.items():
        log(f"Saving interaction outputs for target: {target_name}")
        target_suffix = sanitize_name(target_name)
        csv_path, json_path, plot_path = get_output_paths(output_dir, target_suffix)
        mean_target_interactions = mean_interaction_values(target_interactions)
        save_interaction_outputs(
            mean_target_interactions,
            features.columns.tolist(),
            analysis_feature_names,
            csv_path,
            json_path,
            plot_path,
            f"shapiq Feature Interactions ({target_name})",
            args.max_edges,
        )

    log("Averaging interaction explanations across targets and rows...")
    mean_interactions = mean_interaction_values(
        average_explanations_across_targets(interactions_by_target)
    )
    all_csv_path, all_json_path, all_plot_path = get_output_paths(
        output_dir,
        "all_metrics",
    )
    save_interaction_outputs(
        mean_interactions,
        features.columns.tolist(),
        analysis_feature_names,
        all_csv_path,
        all_json_path,
        all_plot_path,
        "shapiq Feature Interactions (mean across all targets)",
        args.max_edges,
    )

    log("shapiq interaction calculation finished.")
    log(f"Saved all-target interaction table to {all_csv_path}")
    log(f"Saved all-target interaction JSON to {all_json_path}")
    log(f"Saved all-target interaction graph to {all_plot_path}")


if __name__ == "__main__":
    main()
