import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml.constants import TARGET_COLUMNS
from feature_importance.interpretability_utils import (
    clean_transformed_feature_names,
    get_interpretability_output_dir,
    infer_data_dirs_from_model_path,
    load_features,
    load_model,
    log,
    sanitize_name,
    unwrap_pipeline,
)
from feature_importance.shapiq_feature_interactions import (
    mean_interaction_values,
)


POSITIVE_COLOR = "#ff004f"
NEGATIVE_COLOR = "#008bfb"


FEATURE_NAME_ABBREVIATIONS = {
    "numeric_recall": "NR",
    "source_support_hit": "SSH",
    "source_support_max_similarity": "SSMS",
    "token_f1": "TF1",
    "token_overlap": "TO",
    "token_recall": "TR",
}

EXCLUDED_ANALYSIS_FEATURES = {"chunk_overlap"}
FULL_FEATURE_NAMES = {"chunk-size", "dense-k", "sparse-k"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate shapiq InteractionValues and draw native shapiq network_plot "
            "figures for selected interaction orders."
        )
    )
    parser.add_argument(
        "-m",
        "--model-path",
        required=True,
        type=Path,
        help="Path to a saved model.",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        dest="data_dir_options",
        action="append",
        nargs="+",
        type=Path,
        help=(
            "Path(s) to processed train/test data used to compute interactions. "
            "If omitted, paths are inferred from --model-path."
        ),
    )
    parser.add_argument(
        "--target",
        default="all_metrics",
        help=(
            "Target metric to plot, or all_metrics for the mean across targets. "
            "Defaults to all_metrics."
        ),
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        help=(
            "Target metrics to average before plotting. When passed, this overrides "
            "--target. Example: --targets rouge_l_recall cosine_similarity"
        ),
    )
    parser.add_argument(
        "--orders",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Interaction orders to draw. Defaults to: 1 2.",
    )
    parser.add_argument(
        "--n-interactions",
        type=int,
        default=30,
        help="Maximum number of strongest interactions shown by network_plot.",
    )
    parser.add_argument(
        "--draw-threshold",
        type=float,
        default=0.0,
        help="Minimum absolute interaction value drawn by network_plot.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        help=(
            "Evaluation budget per explained row for shapiq TabularExplainer. "
            "Defaults to 2 ** n_features."
        ),
    )
    parser.add_argument(
        "--imputer",
        choices=["marginal", "baseline", "conditional"],
        default="conditional",
        help=(
            "Imputer passed to shapiq TabularExplainer for every calculation. "
            "Defaults to conditional."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory where plots are saved. Defaults to "
            "results/<dataset>/shapiq_network_plots/."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for the network layout.",
    )
    parser.add_argument(
        "--no-circular-layout",
        action="store_true",
        help="Use spring layout instead of the default circular layout.",
    )

    args = parser.parse_args()
    if any(order not in {1, 2} for order in args.orders):
        parser.error("--orders currently supports only 1 and 2.")

    option_dirs = [
        data_dir
        for data_dir_group in (args.data_dir_options or [])
        for data_dir in data_dir_group
    ]
    args.data_dirs = option_dirs or infer_data_dirs_from_model_path(args.model_path)
    del args.data_dir_options

    if not args.data_dirs:
        parser.error("Could not infer data dirs. Pass at least one --data-dir.")

    return args


def abbreviate_feature_name(feature_name):
    if feature_name in FULL_FEATURE_NAMES:
        return feature_name

    if feature_name in FEATURE_NAME_ABBREVIATIONS:
        return FEATURE_NAME_ABBREVIATIONS[feature_name]

    prefix_rules = [
        ("search_mode_", "SM"),
        ("reranker_", "RR"),
        ("llm_model_", "LLM"),
        ("embedding_model_", "EMB"),
        ("retriever_", "RET"),
    ]
    for prefix, abbreviation in prefix_rules:
        if feature_name.startswith(prefix):
            value = feature_name.removeprefix(prefix)
            return f"{abbreviation}:{value}"

    words = feature_name.replace("-", "_").split("_")
    if len(words) <= 1:
        return feature_name

    return "".join(word[:1].upper() for word in words if word)


def abbreviate_feature_names(feature_names):
    abbreviated = []
    used_names = {}

    for feature_name in feature_names:
        short_name = abbreviate_feature_name(feature_name)
        if short_name in used_names:
            used_names[short_name] += 1
            short_name = f"{short_name}{used_names[short_name]}"
        else:
            used_names[short_name] = 1
        abbreviated.append(short_name)

    return abbreviated


def split_model_and_analysis_features(model_features):
    excluded_features = [
        feature_name
        for feature_name in model_features.columns
        if feature_name.split("__", 1)[-1] in EXCLUDED_ANALYSIS_FEATURES
    ]
    if excluded_features:
        log(f"Excluded from shapiq analysis: {', '.join(excluded_features)}")

    analysis_features = model_features.drop(columns=excluded_features)
    return model_features, analysis_features, excluded_features


def resolve_target_names(args):
    if args.targets:
        target_names = args.targets
    elif args.target == "all_metrics":
        target_names = list(TARGET_COLUMNS)
    else:
        target_names = [args.target]

    missing_targets = [
        target_name
        for target_name in target_names
        if target_name not in TARGET_COLUMNS
    ]
    if missing_targets:
        missing = ", ".join(missing_targets)
        available = ", ".join(TARGET_COLUMNS)
        raise ValueError(f"Unknown target(s): {missing}. Available targets: {available}")

    return target_names


def get_output_dir(data_dirs, output_dir=None):
    if output_dir is not None:
        return Path(output_dir)

    return get_interpretability_output_dir(data_dirs, "shapiq_network_plots")


def get_plot_target_label(args):
    if args.targets:
        return "selected_targets"

    return args.target


def get_plot_title_target_label(args):
    if args.targets:
        return "mean of selected targets"

    return args.target


def get_output_target_suffix(args, target_names):
    if args.targets:
        return "selected"

    if args.target == "all_metrics":
        return "all"

    return sanitize_name(args.target)


def add_network_plot_legends(ax, order):
    order_1_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=POSITIVE_COLOR,
            markeredgecolor="none",
            markersize=12,
            label="positive",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=NEGATIVE_COLOR,
            markeredgecolor="none",
            markersize=12,
            label="negative",
        ),
    ]
    order_1_legend = ax.legend(
        handles=order_1_handles,
        title="Main effects",
        loc="upper right",
    )
    order_1_legend.get_title().set_fontweight("bold")
    ax.add_artist(order_1_legend)

    if order < 2:
        return

    order_2_handles = [
        Line2D(
            [0],
            [0],
            color=POSITIVE_COLOR,
            linewidth=4,
            alpha=0.65,
            label="positive",
        ),
        Line2D(
            [0],
            [0],
            color=NEGATIVE_COLOR,
            linewidth=4,
            alpha=0.65,
            label="negative",
        ),
    ]
    order_2_legend = ax.legend(
        handles=order_2_handles,
        title="Pairwise interactions",
        loc="lower right",
    )
    order_2_legend.get_title().set_fontweight("bold")


def predict_selected_target_mean(model, values, target_indices):
    predictions = np.asarray(model.predict(values))

    if predictions.ndim == 1:
        if target_indices != [0]:
            raise ValueError(
                "The model returned one-dimensional predictions, but multiple "
                "target indices were requested."
            )
        return predictions

    if predictions.ndim != 2:
        raise ValueError(f"Unsupported prediction shape: {predictions.shape}")

    return predictions[:, target_indices].mean(axis=1)


def resolve_target_indices(target_names):
    return [TARGET_COLUMNS.index(target_name) for target_name in target_names]


def compute_tabular_interaction_values(
    model,
    model_features,
    analysis_features,
    target_names,
    random_seed,
    imputer,
    budget=None,
):
    from shapiq.explainer import TabularExplainer

    target_indices = resolve_target_indices(target_names)
    feature_array = analysis_features.to_numpy(dtype=float)
    model_feature_columns = model_features.columns.tolist()
    analysis_feature_columns = analysis_features.columns.tolist()
    model_feature_template = model_features.mean(axis=0).to_numpy(dtype=float)

    def selected_target_model(values):
        values = np.atleast_2d(values)
        values_df = pd.DataFrame(
            np.tile(model_feature_template, (len(values), 1)),
            columns=model_feature_columns,
        )
        values_df.loc[:, analysis_feature_columns] = values
        return predict_selected_target_mean(model, values_df, target_indices)

    log(
        "  Computing with shapiq TabularExplainer for selected target mean: "
        f"{', '.join(target_names)}"
    )
    log(f"  Selected target output indices: {target_indices}")
    budget = budget or 2 ** feature_array.shape[1]
    log(f"  Using TabularExplainer imputer={imputer!r}, budget={budget} per row.")
    explainer = TabularExplainer(
        selected_target_model,
        feature_array,
        imputer=imputer,
        index="k-SII",
        max_order=2,
        random_state=random_seed,
    )
    return mean_interaction_values(
        [
            explainer.explain(row, budget=budget, random_state=random_seed)
            for row in feature_array
        ]
    )


def compute_plot_interaction_values(
    model,
    model_features,
    analysis_features,
    target_names,
    args,
):
    return compute_tabular_interaction_values(
        model,
        model_features,
        analysis_features,
        target_names,
        args.random_seed,
        args.imputer,
        budget=args.budget,
    )


def draw_network_plot(
    interaction_values,
    feature_names,
    output_path,
    title,
    order,
    n_interactions,
    draw_threshold,
    random_seed,
    circular_layout,
):
    from shapiq.plot import network_plot

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_interaction_values = interaction_values.get_n_order(
        min_order=1,
        max_order=order,
    )
    fig, ax = network_plot(
        plot_interaction_values,
        feature_names=feature_names,
        show=False,
        n_interactions=n_interactions,
        draw_threshold=draw_threshold,
        random_seed=random_seed,
        circular_layout=circular_layout,
        size_factor=2.5,
        node_size_scaling=1.5,
        plot_original_nodes=True
    )
    ax.set_title(title)
    add_network_plot_legends(ax, order)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def order_1_importance_to_frame(
    interaction_values,
    feature_names,
    target_label,
    target_names,
):
    """Convert first-order shapiq values into a ranked importance table."""
    rows = []

    if hasattr(interaction_values, "interaction_lookup") and hasattr(
        interaction_values,
        "values",
    ):
        for interaction, value_index in interaction_values.interaction_lookup.items():
            if len(interaction) != 1:
                continue

            feature_index = interaction[0]
            rows.append(
                {
                    "parameter": feature_names[feature_index],
                    "importance": float(interaction_values.values[value_index]),
                }
            )
    else:
        order_1_values = np.asarray(
            interaction_values.get_n_order_values(1),
            dtype=float,
        )
        if order_1_values.ndim != 1:
            raise ValueError(
                "Expected first-order interaction values to be a vector, "
                f"got shape {order_1_values.shape}."
            )

        rows = [
            {
                "parameter": feature_name,
                "importance": float(importance),
            }
            for feature_name, importance in zip(feature_names, order_1_values)
        ]

    columns = [
        "rank",
        "target_label",
        "averaged_targets",
        "parameter",
        "importance",
        "abs_importance",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    importance_df = pd.DataFrame(rows)
    importance_df["abs_importance"] = importance_df["importance"].abs()
    importance_df = importance_df.sort_values(
        "abs_importance",
        ascending=False,
    ).reset_index(drop=True)
    importance_df.insert(0, "rank", np.arange(1, len(importance_df) + 1))
    importance_df.insert(1, "target_label", target_label)
    importance_df.insert(2, "averaged_targets", ", ".join(target_names))
    return importance_df[columns]


def save_order_1_importance(
    interaction_values,
    feature_names,
    output_path,
    target_label,
    target_names,
):
    """Save RAG parameters ranked by first-order shapiq importance."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df = order_1_importance_to_frame(
        interaction_values,
        feature_names,
        target_label,
        target_names,
    )
    importance_df.to_csv(output_path, index=False)
    return importance_df


def main():
    args = parse_args()
    target_names = resolve_target_names(args)

    log("Loading model...")
    model, model_path = load_model(args.model_path)
    log(f"Loaded model from {model_path}")

    log("Loading processed features...")
    features = load_features(args.data_dirs)
    estimator, model_features = unwrap_pipeline(model, features)
    model_features, analysis_features, _ = split_model_and_analysis_features(
        model_features
    )
    display_features = clean_transformed_feature_names(analysis_features)

    log(
        f"Loaded {len(features)} rows, {len(model_features.columns)} model features, "
        f"and {len(analysis_features.columns)} analysis features. "
        "Computing shapiq values up to order 2."
    )

    interaction_values = compute_plot_interaction_values(
        estimator,
        model_features,
        analysis_features,
        target_names,
        args,
    )

    output_dir = get_output_dir(args.data_dirs, args.output_dir)
    target_suffix = get_output_target_suffix(args, target_names)
    title_target = get_plot_title_target_label(args)
    feature_names = display_features.columns.tolist()
    plot_feature_names = abbreviate_feature_names(feature_names)
    log(f"network_plot receives {len(plot_feature_names)} abbreviated feature names.")

    importance_path = output_dir / f"order_1_importance_{target_suffix}.csv"
    log(f"Saving order=1 parameter importance table to {importance_path}")
    save_order_1_importance(
        interaction_values,
        feature_names,
        importance_path,
        get_plot_target_label(args),
        target_names,
    )

    for order in args.orders:
        output_path = output_dir / f"network_plot_{target_suffix}_order_{order}.svg"
        title = f"shapiq network_plot ({title_target}, order={order})"
        log(f"Drawing order={order} network_plot to {output_path}")
        draw_network_plot(
            interaction_values,
            plot_feature_names,
            output_path,
            title,
            order,
            args.n_interactions,
            args.draw_threshold,
            args.random_seed,
            circular_layout=not args.no_circular_layout,
        )

    log(f"Saved shapiq network plots to {output_dir}")


if __name__ == "__main__":
    main()
