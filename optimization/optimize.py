import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import optuna
from optuna.trial import TrialState

try:
    from .surrogate import (
        DEFAULT_TARGET_COLUMNS,
        SurrogatePredictor,
        load_summary_features,
    )
except ImportError:
    from surrogate import (
        DEFAULT_TARGET_COLUMNS,
        SurrogatePredictor,
        load_summary_features,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data" / "db" / "summary.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "optimization" / "experiments"
PARAMETER_SPACES_DIR = Path(__file__).resolve().parent / "parameter_spaces"
DEFAULT_OBJECTIVE_METRICS = ["golden_doi_recall", "scientific_fact_recall"]
DEFAULT_BEST_METRIC_FILTERS = {
    "bert_score_recall": 0.3660774008,
    "cosine_similarity": 0.4597832906,
    "golden_doi_mrr": 0.5782741853,
    "rouge_l_recall": 0.5914758181,
}
TOP_CONFIGS_LIMIT = 10
CHUNK_OVERLAP_RATIO = 0.25


def resolve_parameter_space_path(path=None, target_dataset=None):
    """Resolve explicit, dataset-specific, or default parameter space JSON."""
    if path:
        return Path(path)

    if target_dataset:
        dataset_path = PARAMETER_SPACES_DIR / f"parameter_space_{target_dataset}.json"
        if dataset_path.exists():
            return dataset_path

    return PARAMETER_SPACES_DIR / "parameter_space_default.json"


def load_parameter_space(path):
    """Load discrete RAG parameter values from JSON."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as config_file:
        parameter_space = json.load(config_file)

    parameter_space = normalize_parameter_space(parameter_space)
    empty_parameters = [
        name for name, config in parameter_space.items() if not config["values"]
    ]
    if empty_parameters:
        raise ValueError(
            "Each optimized parameter must have a non-empty list of values. "
            f"Invalid parameters: {empty_parameters}"
        )

    return parameter_space


def normalize_parameter_space(parameter_space):
    """Support both list values and detailed per-parameter optimization config."""
    normalized = {}

    for name, config in parameter_space.items():
        if isinstance(config, list):
            values = config
            default = values[0] if values else None
            sampling_weight = 1.0
        elif isinstance(config, dict):
            values = config.get("values", [])
            default = config.get("default", values[0] if values else None)
            sampling_weight = float(config.get("sampling_weight", 1.0))
        else:
            raise ValueError(
                f"Parameter '{name}' must be a list or an object with a 'values' list"
            )

        if default not in values and values:
            raise ValueError(
                f"Default value for parameter '{name}' must be present in its values"
            )

        normalized[name] = {
            "values": values,
            "default": default,
            "sampling_weight": min(max(sampling_weight, 0.0), 1.0),
        }

    return normalized


def validate_objective_metrics(metrics, target_columns):
    unknown_metrics = sorted(set(metrics) - set(target_columns))
    if unknown_metrics:
        raise ValueError(
            f"Objective metrics are not present in target columns: {unknown_metrics}"
        )


def validate_metric_filters(metric_filters, target_columns):
    unknown_metrics = sorted(set(metric_filters) - set(target_columns))
    if unknown_metrics:
        raise ValueError(
            f"Filter metrics are not present in target columns: {unknown_metrics}"
        )


def normalize_weights(metrics, weights=None):
    if weights is None:
        return {metric: 1.0 for metric in metrics}

    if len(weights) != len(metrics):
        raise ValueError("--metric-weights must contain one value per objective metric")

    return dict(zip(metrics, [float(weight) for weight in weights]))


def score_metrics(predicted_metrics, objective_metrics, metric_weights):
    weighted_sum = sum(
        predicted_metrics[metric] * metric_weights[metric]
        for metric in objective_metrics
    )
    weight_sum = sum(abs(metric_weights[metric]) for metric in objective_metrics)
    if weight_sum == 0:
        raise ValueError("At least one metric weight must be non-zero")

    return float(weighted_sum / weight_sum)


def should_vary_parameter(parameter_name, trial_number, seed, sampling_weight):
    if sampling_weight >= 1:
        return True
    if sampling_weight <= 0:
        return False

    digest = hashlib.sha256(
        f"{seed}:{trial_number}:{parameter_name}".encode("utf-8")
    ).hexdigest()
    draw = int(digest[:16], 16) / float(16**16 - 1)
    return draw < sampling_weight


def suggest_parameter(trial, parameter_name, config, seed):
    if should_vary_parameter(
        parameter_name=parameter_name,
        trial_number=trial.number,
        seed=seed,
        sampling_weight=config["sampling_weight"],
    ):
        return trial.suggest_categorical(parameter_name, config["values"])

    return config["default"]


def suggest_parameters(trial, parameter_space, seed):
    params = {}

    if "search_mode" in parameter_space:
        params["search_mode"] = suggest_parameter(
            trial=trial,
            parameter_name="search_mode",
            config=parameter_space["search_mode"],
            seed=seed,
        )

    for name, config in parameter_space.items():
        if name in params:
            continue
        if name == "dense_k" and params.get("search_mode") == "sparse-only":
            params[name] = 0
            continue
        if name == "sparse_k" and params.get("search_mode") == "dense-only":
            params[name] = 0
            continue

        params[name] = suggest_parameter(
            trial=trial,
            parameter_name=name,
            config=config,
            seed=seed,
        )

    if "chunk_size" in params and "chunk_overlap" not in params:
        params["chunk_overlap"] = int(round(params["chunk_size"] * CHUNK_OVERLAP_RATIO))

    return params


def optimize_parameters(
    predictor,
    parameter_space,
    objective_metrics=None,
    metric_weights=None,
    n_trials=100,
    seed=42,
    study_name=None,
):
    objective_metrics = list(objective_metrics or DEFAULT_OBJECTIVE_METRICS)
    validate_objective_metrics(objective_metrics, predictor.target_columns)
    metric_weights = normalize_weights(objective_metrics, metric_weights)

    def objective(trial):
        params = suggest_parameters(trial, parameter_space, seed=seed)
        predicted_metrics = predictor.predict(params)
        objective_value = score_metrics(
            predicted_metrics=predicted_metrics,
            objective_metrics=objective_metrics,
            metric_weights=metric_weights,
        )

        for parameter_name, parameter_value in params.items():
            trial.set_user_attr(f"param_{parameter_name}", parameter_value)

        for metric_name, metric_value in predicted_metrics.items():
            trial.set_user_attr(metric_name, metric_value)

        return objective_value

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials)

    return study


def passes_metric_filters(trial, metric_filters):
    return all(
        trial.user_attrs.get(metric_name, float("-inf")) >= threshold
        for metric_name, threshold in metric_filters.items()
    )


def get_matching_trials(study, metric_filters=None):
    matching_trials = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
        and trial.value is not None
    ]

    if metric_filters:
        matching_trials = [
            trial for trial in matching_trials
            if passes_metric_filters(trial, metric_filters)
        ]

    return matching_trials


def select_best_trial(study, metric_filters=None):
    if not metric_filters:
        return study.best_trial

    matching_trials = get_matching_trials(study, metric_filters=metric_filters)
    if matching_trials:
        return max(matching_trials, key=lambda trial: trial.value)

    raise ValueError(
        "No completed Optuna trials passed the metric filters: "
        + ", ".join(
            f"{metric}>={threshold}" for metric, threshold in metric_filters.items()
        )
    )


def get_trial_params(trial):
    return {
        key.removeprefix("param_"): value
        for key, value in trial.user_attrs.items()
        if key.startswith("param_")
    }


def get_trial_metrics(trial):
    return {
        key: value
        for key, value in trial.user_attrs.items()
        if not key.startswith("param_") and isinstance(value, (int, float))
    }


def make_trial_payload(trial, rank=None):
    payload = {
        "objective_value": float(trial.value),
        "params": get_trial_params(trial),
        "optuna_sampled_params": trial.params,
        "predicted_metrics": get_trial_metrics(trial),
        "trial_number": trial.number,
    }
    if rank is not None:
        payload = {"rank": rank, **payload}

    return payload


def select_top_trials(study, metric_filters=None, limit=TOP_CONFIGS_LIMIT):
    ranked_trials = sorted(
        get_matching_trials(study, metric_filters=metric_filters),
        key=lambda trial: trial.value,
        reverse=True,
    )
    selected_trials = []
    seen_configs = set()

    for trial in ranked_trials:
        config_key = json.dumps(get_trial_params(trial), sort_keys=True)
        if config_key in seen_configs:
            continue

        seen_configs.add(config_key)
        selected_trials.append(trial)
        if len(selected_trials) >= limit:
            break

    return selected_trials


def save_optimization_results(
    study,
    output_dir,
    objective_metrics,
    metric_weights,
    metric_filters=None,
    model_path=None,
    target_dataset=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials_path = output_dir / "optimization_trials.csv"
    best_params_path = output_dir / "best_parameters.json"
    top_params_path = output_dir / "top_10_parameters.json"

    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials.to_csv(trials_path, index=False)

    best_trial = select_best_trial(study, metric_filters=metric_filters)
    best_payload = {
        **make_trial_payload(best_trial),
        "objective_metrics": list(objective_metrics),
        "metric_weights": metric_weights,
        "metric_filters": metric_filters or {},
        "model_path": str(model_path) if model_path else None,
        "target_dataset": target_dataset,
    }
    top_payload = {
        "objective_metrics": list(objective_metrics),
        "metric_weights": metric_weights,
        "metric_filters": metric_filters or {},
        "model_path": str(model_path) if model_path else None,
        "target_dataset": target_dataset,
        "configs": [
            make_trial_payload(trial, rank=index)
            for index, trial in enumerate(
                select_top_trials(study, metric_filters=metric_filters),
                start=1,
            )
        ],
    }

    with best_params_path.open("w", encoding="utf-8") as result_file:
        json.dump(best_payload, result_file, indent=2, ensure_ascii=False)

    with top_params_path.open("w", encoding="utf-8") as result_file:
        json.dump(top_payload, result_file, indent=2, ensure_ascii=False)

    return trials_path, best_params_path, best_trial, top_params_path


def infer_source_dataset_from_model_path(model_path):
    """Infer source dataset from paths like results/<dataset>/hp_tuning/model.pkl."""
    model_path = Path(model_path)
    parts = list(model_path.parts)
    if "results" not in parts:
        return model_path.stem

    results_index = parts.index("results")
    dataset_index = results_index + 1
    if dataset_index >= len(parts):
        return model_path.stem

    return parts[dataset_index]


def make_slug(value):
    """Make a compact filesystem-friendly name segment."""
    allowed_chars = []
    for char in str(value).lower():
        if char.isalnum():
            allowed_chars.append(char)
        elif char in {"-", "_"}:
            allowed_chars.append(char)
        else:
            allowed_chars.append("_")

    slug = "".join(allowed_chars).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")

    return slug or "unknown"


def build_experiment_name(model_path, target_dataset, objective_metrics):
    source_dataset = infer_source_dataset_from_model_path(model_path)
    target_name = target_dataset or "no_target_dataset"
    metrics_name = "_".join(make_slug(metric) for metric in objective_metrics)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return (
        f"{make_slug(source_dataset)}_to_{make_slug(target_name)}"
        f"__{metrics_name}__{timestamp}"
    )


def resolve_experiment_dir(output_root, experiment_name, model_path, target_dataset, metrics):
    experiment_name = experiment_name or build_experiment_name(
        model_path=model_path,
        target_dataset=target_dataset,
        objective_metrics=metrics,
    )
    return Path(output_root) / experiment_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize discrete RAG parameters with Optuna and surrogate models."
    )
    parser.add_argument(
        "-m",
        "--model-path",
        required=True,
        type=Path,
        help="Path to a saved surrogate model pickle.",
    )
    parser.add_argument(
        "-p",
        "--parameter-space",
        type=Path,
        help=(
            "Path to JSON with discrete parameter values. If omitted, the optimizer "
            "uses optimization/parameter_spaces/parameter_space_<target_dataset>.json "
            "when available, otherwise parameter_space_default.json."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_OBJECTIVE_METRICS,
        help="Metrics to maximize.",
    )
    parser.add_argument(
        "--metric-weights",
        nargs="+",
        type=float,
        help="Optional weights for objective metrics.",
    )
    parser.add_argument(
        "--filter-best-metrics",
        action="store_true",
        help=(
            "Select the final best configuration only among trials whose predicted "
            "bert_score_recall, cosine_similarity, golden_doi_mrr, and "
            "rouge_l_recall meet the configured default thresholds."
        ),
    )
    parser.add_argument(
        "--target-columns",
        nargs="+",
        default=DEFAULT_TARGET_COLUMNS,
        help="Target names in the same order as surrogate predictions.",
    )
    parser.add_argument(
        "--target-dataset",
        help=(
            "Dataset whose summary features are passed to the surrogate. "
            "Use this for transfer optimization, for example model trained on Y "
            "and parameters optimized for X."
        ),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Path to dataset summary CSV. Defaults to {DEFAULT_SUMMARY_PATH}",
    )
    parser.add_argument(
        "-n",
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Optuna sampler.",
    )
    parser.add_argument(
        "--study-name",
        help="Optional Optuna study name.",
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for optimization experiments. Defaults to {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--experiment-name",
        help="Optional experiment folder name inside --output-root.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    parameter_space_path = resolve_parameter_space_path(
        path=args.parameter_space,
        target_dataset=args.target_dataset,
    )
    parameter_space = load_parameter_space(parameter_space_path)
    extra_features = load_summary_features(args.target_dataset, args.summary_path)
    predictor = SurrogatePredictor.from_pickle(
        model_path=args.model_path,
        target_columns=args.target_columns,
        extra_features=extra_features,
    )
    metric_filters = DEFAULT_BEST_METRIC_FILTERS if args.filter_best_metrics else None
    if metric_filters:
        validate_metric_filters(metric_filters, predictor.target_columns)
    study = optimize_parameters(
        predictor=predictor,
        parameter_space=parameter_space,
        objective_metrics=args.metrics,
        metric_weights=args.metric_weights,
        n_trials=args.n_trials,
        seed=args.seed,
        study_name=args.study_name,
    )
    metric_weights = normalize_weights(args.metrics, args.metric_weights)
    experiment_dir = resolve_experiment_dir(
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        model_path=args.model_path,
        target_dataset=args.target_dataset,
        metrics=args.metrics,
    )
    (
        trials_path,
        best_params_path,
        best_trial,
        top_params_path,
    ) = save_optimization_results(
        study=study,
        output_dir=experiment_dir,
        objective_metrics=args.metrics,
        metric_weights=metric_weights,
        metric_filters=metric_filters,
        model_path=args.model_path,
        target_dataset=args.target_dataset,
    )

    print("Best objective value:", f"{best_trial.value:.6f}")
    print("Best parameters:")
    best_params = {
        key.removeprefix("param_"): value
        for key, value in best_trial.user_attrs.items()
        if key.startswith("param_")
    }
    print(json.dumps(best_params, indent=2, ensure_ascii=False))
    if metric_filters:
        print("Applied best-trial metric filters:")
        print(json.dumps(metric_filters, indent=2, ensure_ascii=False))
    if args.target_dataset:
        print(f"Optimized for target dataset: {args.target_dataset}")
    print(f"Loaded surrogate model from {args.model_path}")
    print(f"Loaded parameter space from {parameter_space_path}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Saved trials to {trials_path}")
    print(f"Saved best parameters to {best_params_path}")
    print(f"Saved top-10 parameters to {top_params_path}")


if __name__ == "__main__":
    main()
