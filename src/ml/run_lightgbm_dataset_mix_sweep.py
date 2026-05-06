import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from matplotlib.ticker import MultipleLocator

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml.constants import ML_RESULTS_DIR, TRAIN_MODELS_RANDOM_STATE
from ml.evaluate import metrics_by_target
from ml.hyperparameter_tuning import build_estimator, get_param_grid, tune_model_grid
from ml.train_models import (
    build_model_pipeline,
    evaluate_model,
    get_feature_types,
    load_processed_frames,
    split_features_targets,
)


PLOT_TARGETS = [
    "bert_score_recall",
    "cosine_similarity",
    "golden_doi_mrr",
    "golden_doi_recall",
    "rouge_l_recall",
    "scientific_fact_recall",
]

TUNING_TARGETS = [
    "golden_doi_recall",
    "scientific_fact_recall",
    "cosine_similarity",
    "bert_score_recall",
    "rouge_l_recall",
    "golden_doi_mrr",
]

TARGET_R2_TUNING_THRESHOLD = 0.8


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train LightGBM models on the full first processed dataset plus "
            "a varying share of the second train split, then test on the "
            "second dataset."
        )
    )
    parser.add_argument(
        "first_data_dir",
        type=Path,
        help="First processed dataset directory with train.csv and test.csv.",
    )
    parser.add_argument(
        "second_data_dir",
        type=Path,
        help="Second processed dataset directory with train.csv and test.csv.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=TRAIN_MODELS_RANDOM_STATE,
        help="Base random state used for sampling rows from each split.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help=(
            "Directory to save models, metrics, and plot. Defaults to "
            "results/<first>_plus_<second>_lightgbm_sweep."
        ),
    )
    parser.add_argument(
        "--tune-full-mix-only",
        action="store_true",
        help=(
            "Tune LightGBM hyperparameters only for the model trained on "
            "100%% of the first dataset plus 100%% of the second dataset. "
            "When set, the R2 threshold-based tuning selection is skipped."
        ),
    )
    return parser.parse_args()


def sample_rows(data, row_count, random_state):
    """Sample exactly row_count rows without replacement."""
    if row_count > len(data):
        raise ValueError(
            f"Cannot sample {row_count} rows from dataset with {len(data)} rows"
        )

    return data.sample(n=row_count, random_state=random_state).copy()


def build_augmented_split(first_data, second_data, second_fraction, random_state):
    """Use all first_data rows and add a sampled share of second_data rows."""
    second_count = round(len(second_data) * second_fraction)
    frames = [first_data.copy()]

    if second_count:
        frames.append(
            sample_rows(
                second_data,
                second_count,
                random_state=random_state,
            )
        )

    augmented = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    return augmented.sample(frac=1, random_state=random_state + 1).reset_index(drop=True)


def train_lightgbm_model(x_train, y_train, random_state):
    """Train a LightGBM multi-output model with shared preprocessing."""
    numeric_features, categorical_features = get_feature_types(x_train)
    model = build_model_pipeline(
        MultiOutputRegressor(LGBMRegressor(random_state=random_state, verbosity=-1)),
        numeric_features,
        categorical_features,
    )
    model.fit(x_train, y_train)
    return model


def tune_lightgbm_model(x_train, y_train, random_state):
    """Tune LightGBM with the same grid used by hyperparameter_tuning.py."""
    numeric_features, categorical_features = get_feature_types(x_train)
    estimator = build_estimator(
        "LightGBM",
        numeric_features,
        categorical_features,
        random_state=random_state,
    )
    search = tune_model_grid(
        estimator,
        get_param_grid("LightGBM"),
        x_train,
        y_train,
    )
    return search


def add_run_columns(metrics_df, run_metadata):
    """Attach mixture metadata columns to a metrics dataframe."""
    metrics_df = metrics_df.copy()
    for column, value in reversed(list(run_metadata.items())):
        metrics_df.insert(0, column, value)
    return metrics_df


def save_model(model, output_dir, run_name, run_metadata, feature_names):
    """Save a trained model bundle."""
    model_path = output_dir / "models" / f"{run_name}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": model,
        "random_state": run_metadata["random_state"],
        "mixture": run_metadata,
        "preprocessing": {
            "feature_names": feature_names,
        },
    }
    with model_path.open("wb") as model_file:
        pickle.dump(model_bundle, model_file)

    return model_path


def save_tuned_model(
    search,
    output_dir,
    run_name,
    run_metadata,
    feature_names,
    tuning_mode,
):
    """Save the best tuned LightGBM model bundle for the selected mixture."""
    model_path = output_dir / "models" / "tuned" / tuning_mode / f"{run_name}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": search.best_estimator_,
        "random_state": run_metadata["random_state"],
        "mixture": run_metadata,
        "tuning": {
            "method": "grid",
            "mode": tuning_mode,
            "model": "LightGBM",
            "target_r2_threshold": TARGET_R2_TUNING_THRESHOLD,
            "threshold_targets": TUNING_TARGETS,
            "best_cv_rmse": float(-search.best_score_),
            "best_params": search.best_params_,
        },
        "preprocessing": {
            "feature_names": feature_names,
        },
    }
    with model_path.open("wb") as model_file:
        pickle.dump(model_bundle, model_file)

    return model_path


def find_first_threshold_mix(target_metrics_df):
    """Return the first mix where all selected target R2 values reach threshold."""
    missing_targets = sorted(
        set(TUNING_TARGETS) - set(target_metrics_df["target"].unique())
    )
    if missing_targets:
        raise ValueError(
            "Cannot select mix for tuning because target metrics are missing: "
            f"{missing_targets}"
        )

    selected = target_metrics_df[target_metrics_df["target"].isin(TUNING_TARGETS)]
    target_r2_by_percent = selected.pivot_table(
        index="second_percent",
        columns="target",
        values="r2",
        aggfunc="first",
    )
    target_r2_by_percent = target_r2_by_percent.sort_index()
    passing = target_r2_by_percent[
        target_r2_by_percent.ge(TARGET_R2_TUNING_THRESHOLD).all(axis=1)
    ]

    if passing.empty:
        return None

    return int(passing.index[0])


def build_tuning_summary_row(
    search,
    metrics,
    target_metrics,
    model_path,
    run_metadata,
    tuning_mode,
):
    """Create a one-row summary for the tuned threshold model."""
    return {
        **run_metadata,
        "model_path": str(model_path),
        "method": "grid",
        "tuning_mode": tuning_mode,
        "model": "LightGBM",
        "threshold": TARGET_R2_TUNING_THRESHOLD,
        "threshold_targets": json.dumps(TUNING_TARGETS),
        "best_cv_rmse": float(-search.best_score_),
        "best_params": json.dumps(search.best_params_),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "threshold_targets_min_r2": float(
            target_metrics[target_metrics["target"].isin(TUNING_TARGETS)]["r2"].min()
        ),
    }


def resolve_tuning_percent(args, target_metrics_df):
    """Choose which dataset mix should be tuned."""
    if args.tune_full_mix_only:
        return 100, "full_mix"

    threshold_percent = find_first_threshold_mix(target_metrics_df)
    if threshold_percent is None:
        return None, "threshold"

    return threshold_percent, "threshold"


def plot_r2_by_mix(target_metrics_df, first_name, second_name, output_path):
    """Plot target-level R2 over dataset composition."""
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 7))

    plot_df = target_metrics_df[target_metrics_df["target"].isin(PLOT_TARGETS)].copy()
    if plot_df.empty:
        raise ValueError(f"No plot targets found in target metrics: {PLOT_TARGETS}")

    plot_df = plot_df.sort_values(["target", "second_percent"])
    x_values = sorted(plot_df["second_percent"].unique())
    x_positions = list(range(len(x_values)))
    x_position_by_value = dict(zip(x_values, x_positions))

    names = {"complexes": "MC", "nanozymes": "NZ", "oxazo": "OA"}

    x_labels = [f"{value}%" for value in x_values]

    for target, target_points in plot_df.groupby("target", sort=True):
        target_points = target_points.sort_values("second_percent")
        ax.plot(
            [x_position_by_value[value] for value in target_points["second_percent"]],
            target_points["r2"],
            marker="o",
            linewidth=2,
            markersize=5,
            label=target,
        )

    ax.set_title(
        f"full {first_name} train plus sampled {second_name} train (LightGBM)"
    )
    ax.set_xlabel("Added train share")
    ax.set_ylabel("R2")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x_positions, x_labels)
    ax.set_xlim(-0.2, len(x_positions) - 0.8 if len(x_positions) > 1 else 0.2)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.legend(
        title="Target",
        loc="lower right",
    )
    # plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=500, format="svg")
    plt.close(fig)


def get_default_output_dir(first_data_dir, second_data_dir):
    first_name = Path(first_data_dir).name
    second_name = Path(second_data_dir).name
    return ML_RESULTS_DIR / f"{first_name}_plus_{second_name}_lightgbm_sweep"


def main():
    args = parse_args()
    first_name = args.first_data_dir.name
    second_name = args.second_data_dir.name
    output_dir = args.output_dir or get_default_output_dir(
        args.first_data_dir,
        args.second_data_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    first_train, _ = load_processed_frames(args.first_data_dir)
    second_train, second_test = load_processed_frames(args.second_data_dir)

    metric_rows = []
    target_metric_frames = []
    run_data_by_percent = {}
    fractions = [percent / 100 for percent in range(0, 101, 10)]

    for run_index, second_fraction in enumerate(fractions, start=1):
        second_percent = round(second_fraction * 100)
        run_name = f"{first_name}_100_{second_name}_{second_percent}"
        run_random_state = args.random_state

        print(f"Training {run_name}...")
        mixed_train = build_augmented_split(
            first_train,
            second_train,
            second_fraction=second_fraction,
            random_state=run_random_state,
        )
        x_train, y_train = split_features_targets(mixed_train)
        x_test, y_test = split_features_targets(second_test)
        x_train, x_test = x_train.align(x_test, join="outer", axis=1, fill_value=0)
        run_data_by_percent[second_percent] = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "run_metadata": None,
        }

        model = train_lightgbm_model(x_train, y_train, random_state=run_random_state)
        metrics = evaluate_model(model, x_test, y_test)
        run_metadata = {
            "run": run_index,
            "run_name": run_name,
            "first_dataset": first_name,
            "second_dataset": second_name,
            "first_percent": 100,
            "second_percent": second_percent,
            "second_train_rows": len(mixed_train) - len(first_train),
            "second_test_rows": len(second_test),
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "random_state": run_random_state,
        }
        run_data_by_percent[second_percent]["run_metadata"] = run_metadata
        model_path = save_model(
            model,
            output_dir,
            run_name,
            run_metadata,
            feature_names=x_train.columns.tolist(),
        )

        metric_rows.append(
            {
                **run_metadata,
                "model_path": str(model_path),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
            }
        )
        target_metrics = metrics_by_target(metrics)
        target_metrics.insert(0, "model_path", str(model_path))
        target_metrics = add_run_columns(target_metrics, run_metadata)
        target_metric_frames.append(target_metrics)

    metrics_df = pd.DataFrame(metric_rows)
    target_metrics_df = pd.concat(target_metric_frames, ignore_index=True)

    metrics_path = output_dir / "lightgbm_mix_metrics.csv"
    target_metrics_path = output_dir / "lightgbm_mix_target_metrics.csv"
    plot_path = output_dir / "r2_by_dataset_mix.svg"

    metrics_df.to_csv(metrics_path, index=False)
    target_metrics_df.to_csv(target_metrics_path, index=False)

    tuning_percent, tuning_mode = resolve_tuning_percent(args, target_metrics_df)
    tuning_summary_path = output_dir / f"lightgbm_{tuning_mode}_tuned_model_metrics.csv"
    tuning_target_metrics_path = (
        output_dir / f"lightgbm_{tuning_mode}_tuned_target_metrics.csv"
    )
    if tuning_percent is None:
        print(
            "No dataset mix reached "
            f"R2 >= {TARGET_R2_TUNING_THRESHOLD} for all threshold targets; "
            "skipping LightGBM hyperparameter tuning."
        )
    else:
        tuned_run = run_data_by_percent[tuning_percent]
        run_metadata = tuned_run["run_metadata"].copy()
        run_name = run_metadata["run_name"]

        if tuning_mode == "full_mix":
            print(f"Tuning LightGBM for full 100% + 100% mix {run_name}...")
        else:
            print(
                "Tuning LightGBM for threshold mix "
                f"{run_name} "
                f"(all selected target R2 >= {TARGET_R2_TUNING_THRESHOLD})..."
            )
        search = tune_lightgbm_model(
            tuned_run["x_train"],
            tuned_run["y_train"],
            random_state=run_metadata["random_state"],
        )
        tuned_model_path = save_tuned_model(
            search,
            output_dir,
            run_name,
            run_metadata,
            feature_names=tuned_run["x_train"].columns.tolist(),
            tuning_mode=tuning_mode,
        )
        tuned_metrics = evaluate_model(
            search.best_estimator_,
            tuned_run["x_test"],
            tuned_run["y_test"],
        )
        tuned_target_metrics = metrics_by_target(tuned_metrics)
        tuned_target_metrics.insert(0, "model_path", str(tuned_model_path))
        tuned_target_metrics = add_run_columns(tuned_target_metrics, run_metadata)
        tuned_summary = pd.DataFrame(
            [
                build_tuning_summary_row(
                    search,
                    tuned_metrics,
                    tuned_target_metrics,
                    tuned_model_path,
                    run_metadata,
                    tuning_mode,
                )
            ]
        )

        tuned_summary.to_csv(tuning_summary_path, index=False)
        tuned_target_metrics.to_csv(tuning_target_metrics_path, index=False)

    plot_r2_by_mix(target_metrics_df, first_name, second_name, plot_path)

    print(f"Saved models to {output_dir / 'models'}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved target metrics to {target_metrics_path}")
    if tuning_percent is not None:
        print(f"Saved {tuning_mode} tuned model metrics to {tuning_summary_path}")
        print(
            f"Saved {tuning_mode} tuned target metrics to {tuning_target_metrics_path}"
        )
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
