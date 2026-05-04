import argparse
import hashlib
import json
import math
import time
from itertools import product
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
BEST_METRIC_FILTERS_BY_TARGET_DATASET = {
    "oxazo": {
        "bert_score_recall": 0.3769322453,
        "cosine_similarity": 0.5352906687,
        "golden_doi_mrr": 0.5916038563,
        "rouge_l_recall": 0.6211845608,
    },
    "nanozymes": {
        "bert_score_recall": 0.4380627381,
        "cosine_similarity": 0.4781967146,
        "golden_doi_mrr": 0.579405288,
        "rouge_l_recall": 0.7046844593,
    },
    "nano": {
        "bert_score_recall": 0.4380627381,
        "cosine_similarity": 0.4781967146,
        "golden_doi_mrr": 0.579405288,
        "rouge_l_recall": 0.7046844593,
    },
    "complexes": {
        "bert_score_recall": 0.3925981267,
        "cosine_similarity": 0.5450280985,
        "golden_doi_mrr": 0.5908610546,
        "rouge_l_recall": 0.6153255168,
    },
}
TOP10_METRIC_FILTERS_BY_TARGET_DATASET = {
    "nano": {
        "bert_score_recall": 0.437436934,
        "cosine_similarity": 0.4597832906,
        "golden_doi_mrr": 0.5782741853,
        "rouge_l_recall": 0.6850525277,
    },
    "nanozymes": {
        "bert_score_recall": 0.437436934,
        "cosine_similarity": 0.4597832906,
        "golden_doi_mrr": 0.5782741853,
        "rouge_l_recall": 0.6850525277,
    },
    "oxazo": {
        "bert_score_recall": 0.3660774008,
        "cosine_similarity": 0.4877119864,
        "golden_doi_mrr": 0.5811471901,
        "rouge_l_recall": 0.6046754656,
    },
    "complexes": {
        "bert_score_recall": 0.3879239623,
        "cosine_similarity": 0.515907508,
        "golden_doi_mrr": 0.5817711655,
        "rouge_l_recall": 0.5914758181,
    },
}
CHUNK_OVERLAP_RATIO = 0.25
OBJECTIVE_TIE_ABS_TOL = 1e-12
CHEAP_NUMERIC_PARAMS = ("chunk_size", "dense_k", "sparse_k")
KNEE_SCORE_ABS_TOL = 1e-12


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


def resolve_best_metric_filters(target_dataset=None):
    if not target_dataset:
        return DEFAULT_BEST_METRIC_FILTERS

    return BEST_METRIC_FILTERS_BY_TARGET_DATASET.get(
        target_dataset,
        DEFAULT_BEST_METRIC_FILTERS,
    )


def resolve_top10_metric_filters(target_dataset=None):
    if not target_dataset:
        return DEFAULT_BEST_METRIC_FILTERS

    return TOP10_METRIC_FILTERS_BY_TARGET_DATASET.get(
        target_dataset,
        DEFAULT_BEST_METRIC_FILTERS,
    )


def format_duration(seconds):
    seconds = float(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {seconds:.2f}s"

    return f"{seconds:.2f}s"


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
    multi_objective=False,
):
    objective_metrics = list(objective_metrics or DEFAULT_OBJECTIVE_METRICS)
    validate_objective_metrics(objective_metrics, predictor.target_columns)
    metric_weights = normalize_weights(objective_metrics, metric_weights)

    def objective(trial):
        params = suggest_parameters(trial, parameter_space, seed=seed)
        predicted_metrics = predictor.predict(params)
        for parameter_name, parameter_value in params.items():
            trial.set_user_attr(f"param_{parameter_name}", parameter_value)

        for metric_name, metric_value in predicted_metrics.items():
            trial.set_user_attr(metric_name, metric_value)

        if multi_objective:
            return tuple(predicted_metrics[metric] for metric in objective_metrics)

        objective_value = score_metrics(
            predicted_metrics=predicted_metrics,
            objective_metrics=objective_metrics,
            metric_weights=metric_weights,
        )
        return objective_value

    sampler = optuna.samplers.TPESampler(seed=seed)
    study_config = {
        "sampler": sampler,
        "study_name": study_name,
    }
    if multi_objective:
        study_config["directions"] = ["maximize"] * len(objective_metrics)
    else:
        study_config["direction"] = "maximize"

    study = optuna.create_study(**study_config)
    started_at = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    study.optimization_duration_seconds = time.perf_counter() - started_at

    return study


def passes_metric_filters(trial, metric_filters):
    return all(
        trial.user_attrs.get(metric_name, float("-inf")) >= threshold
        for metric_name, threshold in metric_filters.items()
    )


def trial_has_objective(trial):
    values = getattr(trial, "values", None)
    if values is not None:
        return True

    return trial.value is not None


def is_multi_objective_study(study):
    return len(getattr(study, "directions", [])) > 1


def get_matching_trials(study, metric_filters=None):
    matching_trials = [
        trial
        for trial in study.trials
        if trial.state == TrialState.COMPLETE
        and trial_has_objective(trial)
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


def params_key(params):
    return json.dumps(params, sort_keys=True)


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


def get_trial_scalar_score(trial, objective_metrics=None, metric_weights=None):
    if objective_metrics:
        metrics = get_trial_metrics(trial)
        return score_metrics(
            predicted_metrics=metrics,
            objective_metrics=objective_metrics,
            metric_weights=metric_weights or normalize_weights(objective_metrics),
        )

    return float(trial.value)


def get_trial_objective_values(trial, objective_metrics=None):
    values = getattr(trial, "values", None)
    if values is None:
        return None

    values = [float(value) for value in values]
    if objective_metrics and len(objective_metrics) == len(values):
        return dict(zip(objective_metrics, values))

    return values


def make_trial_payload(trial, rank=None, objective_metrics=None, metric_weights=None):
    payload = {
        "objective_value": get_trial_scalar_score(
            trial,
            objective_metrics=objective_metrics,
            metric_weights=metric_weights,
        ),
        "params": get_trial_params(trial),
        "optuna_sampled_params": trial.params,
        "predicted_metrics": get_trial_metrics(trial),
        "trial_number": trial.number,
    }
    objective_values = (
        get_trial_objective_values(trial, objective_metrics=objective_metrics)
        if objective_metrics
        else None
    )
    if objective_values is not None:
        payload["objective_values"] = objective_values

    if rank is not None:
        payload = {"rank": rank, **payload}

    return payload


def unique_config_trials(trials):
    config_trials = {}

    for trial in sorted(trials, key=lambda item: item.number):
        config_key = params_key(get_trial_params(trial))
        if config_key in config_trials:
            continue

        config_trials[config_key] = trial

    return list(config_trials.values())


def get_best_score_trials(study, metric_filters=None):
    matching_trials = get_matching_trials(study, metric_filters=metric_filters)
    if not matching_trials:
        raise ValueError(
            "No completed Optuna trials passed the metric filters: "
            + ", ".join(
                f"{metric}>={threshold}" for metric, threshold in (metric_filters or {}).items()
            )
        )

    best_value = max(trial.value for trial in matching_trials)
    return [
        trial
        for trial in matching_trials
        if abs(trial.value - best_value) <= OBJECTIVE_TIE_ABS_TOL
    ]


def cheapness_key(trial):
    params = get_trial_params(trial)
    numeric_key = tuple(float(params.get(name, 0)) for name in CHEAP_NUMERIC_PARAMS)
    return numeric_key + (trial.number,)


def select_cheapest_trial(trials):
    return min(trials, key=cheapness_key)


def get_mode_value_orders(trials):
    params_by_trial = [get_trial_params(trial) for trial in trials]
    parameter_names = list(params_by_trial[0].keys())
    value_orders = {}

    for parameter_name in parameter_names:
        counts = {}
        first_seen = {}

        for index, params in enumerate(params_by_trial):
            value = params[parameter_name]
            key = json.dumps(value, sort_keys=True)
            counts[key] = counts.get(key, 0) + 1
            first_seen.setdefault(key, (index, value))

        ordered = sorted(
            counts,
            key=lambda key: (-counts[key], first_seen[key][0]),
        )
        value_orders[parameter_name] = [first_seen[key][1] for key in ordered]

    return parameter_names, value_orders


def iter_mode_candidate_params(parameter_names, value_orders):
    ranked_candidates = []
    rank_ranges = [
        range(len(value_orders[parameter_name]))
        for parameter_name in parameter_names
    ]

    for ranks in product(*rank_ranges):
        candidate = {
            parameter_name: value_orders[parameter_name][rank]
            for parameter_name, rank in zip(parameter_names, ranks)
        }
        ranked_candidates.append((sum(ranks), ranks, candidate))

    for _, _, candidate in sorted(ranked_candidates, key=lambda item: (item[0], item[1])):
        yield candidate


def select_mode_trial(trials):
    trial_by_config = {params_key(get_trial_params(trial)): trial for trial in trials}
    parameter_names, value_orders = get_mode_value_orders(trials)

    for candidate in iter_mode_candidate_params(parameter_names, value_orders):
        trial = trial_by_config.get(params_key(candidate))
        if trial is not None:
            return trial

    return max(
        trials,
        key=lambda trial: sum(
            get_trial_params(other_trial) == get_trial_params(trial)
            for other_trial in trials
        ),
    )


def get_trial_objective_list(trial, objective_metrics):
    values = getattr(trial, "values", None)
    if values is not None:
        return [float(value) for value in values]

    metrics = get_trial_metrics(trial)
    return [float(metrics[metric]) for metric in objective_metrics]


def dominates(candidate_trial, other_trial, objective_metrics):
    candidate_values = get_trial_objective_list(candidate_trial, objective_metrics)
    other_values = get_trial_objective_list(other_trial, objective_metrics)

    return (
        all(
            candidate_value >= other_value
            for candidate_value, other_value in zip(candidate_values, other_values)
        )
        and any(
            candidate_value > other_value
            for candidate_value, other_value in zip(candidate_values, other_values)
        )
    )


def get_pareto_trials(study, objective_metrics, metric_filters=None):
    matching_trials = get_matching_trials(study, metric_filters=metric_filters)
    if not matching_trials:
        raise ValueError(
            "No completed Optuna trials passed the metric filters: "
            + ", ".join(
                f"{metric}>={threshold}" for metric, threshold in (metric_filters or {}).items()
            )
        )

    return [
        trial
        for trial in matching_trials
        if not any(
            dominates(other_trial, trial, objective_metrics)
            for other_trial in matching_trials
            if other_trial.number != trial.number
        )
    ]


def get_best_scalar_trials(trials, objective_metrics, metric_weights):
    best_score = max(
        get_trial_scalar_score(
            trial,
            objective_metrics=objective_metrics,
            metric_weights=metric_weights,
        )
        for trial in trials
    )
    return [
        trial
        for trial in trials
        if abs(
            get_trial_scalar_score(
                trial,
                objective_metrics=objective_metrics,
                metric_weights=metric_weights,
            )
            - best_score
        )
        <= OBJECTIVE_TIE_ABS_TOL
    ]


def normalize_objective_vectors(trials, objective_metrics):
    vectors = [
        get_trial_objective_list(trial, objective_metrics)
        for trial in trials
    ]
    mins = [min(vector[index] for vector in vectors) for index in range(len(objective_metrics))]
    maxes = [max(vector[index] for vector in vectors) for index in range(len(objective_metrics))]
    normalized = {}

    for trial, vector in zip(trials, vectors):
        normalized[trial.number] = [
            1.0 if max_value == min_value else (value - min_value) / (max_value - min_value)
            for value, min_value, max_value in zip(vector, mins, maxes)
        ]

    return normalized


def euclidean_distance(point, other_point):
    return math.sqrt(
        sum((value - other_value) ** 2 for value, other_value in zip(point, other_point))
    )


def distance_to_line(point, line_start, line_end):
    direction = [
        end_value - start_value
        for start_value, end_value in zip(line_start, line_end)
    ]
    direction_norm_sq = sum(value ** 2 for value in direction)
    if direction_norm_sq == 0:
        return euclidean_distance(point, line_start)

    point_offset = [
        value - start_value
        for value, start_value in zip(point, line_start)
    ]
    projection_scale = sum(
        offset_value * direction_value
        for offset_value, direction_value in zip(point_offset, direction)
    ) / direction_norm_sq
    projection = [
        start_value + projection_scale * direction_value
        for start_value, direction_value in zip(line_start, direction)
    ]

    return euclidean_distance(point, projection)


def get_knee_score_by_trial(trials, objective_metrics):
    if len(trials) == 1:
        return {trials[0].number: 0.0}

    normalized = normalize_objective_vectors(trials, objective_metrics)
    metric_count = len(objective_metrics)
    if metric_count == 1:
        return {
            trial.number: normalized[trial.number][0]
            for trial in trials
        }

    if metric_count == 2:
        ordered_trials = sorted(
            trials,
            key=lambda trial: (normalized[trial.number][0], normalized[trial.number][1]),
        )
        line_start = normalized[ordered_trials[0].number]
        line_end = normalized[ordered_trials[-1].number]
    else:
        line_start = [0.0] * metric_count
        line_end = [1.0] * metric_count

    return {
        trial.number: distance_to_line(normalized[trial.number], line_start, line_end)
        for trial in trials
    }


def get_knee_trials(pareto_trials, objective_metrics):
    knee_scores = get_knee_score_by_trial(pareto_trials, objective_metrics)
    best_knee_score = max(knee_scores.values())

    return [
        trial
        for trial in pareto_trials
        if abs(knee_scores[trial.number] - best_knee_score) <= KNEE_SCORE_ABS_TOL
    ], knee_scores


def make_multi_objective_payload(
    study,
    objective_metrics,
    metric_weights,
    metric_filters=None,
    model_path=None,
    target_dataset=None,
):
    pareto_trials = unique_config_trials(
        get_pareto_trials(
            study,
            objective_metrics=objective_metrics,
            metric_filters=metric_filters,
        )
    )
    knee_trials, knee_scores = get_knee_trials(
        pareto_trials,
        objective_metrics=objective_metrics,
    )
    best_config_trials = unique_config_trials(knee_trials)
    first_best_trial = min(knee_trials, key=lambda trial: trial.number)
    common_payload = {
        "optimization_mode": "multi_objective",
        "objective_metrics": list(objective_metrics),
        "metric_weights": metric_weights,
        "metric_filters": metric_filters or {},
        "model_path": str(model_path) if model_path else None,
        "target_dataset": target_dataset,
        "pareto_selection_method": "knee_point",
        "pareto_unique_configs": len(pareto_trials),
        "knee_unique_configs": len(best_config_trials),
        "pareto_configs": [
            {
                **make_trial_payload(
                    trial,
                    rank=index,
                    objective_metrics=objective_metrics,
                    metric_weights=metric_weights,
                ),
                "knee_score": knee_scores[trial.number],
            }
            for index, trial in enumerate(
                sorted(
                    pareto_trials,
                    key=lambda trial: knee_scores[trial.number],
                    reverse=True,
                ),
                start=1,
            )
        ],
    }

    if len(best_config_trials) == 1:
        return {
            **make_trial_payload(
                first_best_trial,
                objective_metrics=objective_metrics,
                metric_weights=metric_weights,
            ),
            **common_payload,
            "selection_strategy": "pareto_knee_unique_best",
            "knee_score": knee_scores[first_best_trial.number],
        }, first_best_trial

    cheapest_trial = select_cheapest_trial(best_config_trials)
    mode_trial = select_mode_trial(best_config_trials)
    payload = {
        **make_trial_payload(
            first_best_trial,
            objective_metrics=objective_metrics,
            metric_weights=metric_weights,
        ),
        **common_payload,
        "selection_strategy": "pareto_knee_tie_break_strategies",
        "knee_score": knee_scores[first_best_trial.number],
        "selected_strategy": "first_best",
        "best_configs": {
            "first_best": {
                **make_trial_payload(
                    first_best_trial,
                    objective_metrics=objective_metrics,
                    metric_weights=metric_weights,
                ),
                "knee_score": knee_scores[first_best_trial.number],
            },
            "cheapest": {
                **make_trial_payload(
                    cheapest_trial,
                    objective_metrics=objective_metrics,
                    metric_weights=metric_weights,
                ),
                "knee_score": knee_scores[cheapest_trial.number],
            },
            "mode": {
                **make_trial_payload(
                    mode_trial,
                    objective_metrics=objective_metrics,
                    metric_weights=metric_weights,
                ),
                "knee_score": knee_scores[mode_trial.number],
            },
        },
    }

    return payload, first_best_trial


def make_best_payload(
    study,
    objective_metrics,
    metric_weights,
    metric_filters=None,
    model_path=None,
    target_dataset=None,
):
    if is_multi_objective_study(study):
        return make_multi_objective_payload(
            study=study,
            objective_metrics=objective_metrics,
            metric_weights=metric_weights,
            metric_filters=metric_filters,
            model_path=model_path,
            target_dataset=target_dataset,
        )

    best_score_trials = get_best_score_trials(study, metric_filters=metric_filters)
    best_config_trials = unique_config_trials(best_score_trials)
    common_payload = {
        "objective_metrics": list(objective_metrics),
        "metric_weights": metric_weights,
        "metric_filters": metric_filters or {},
        "model_path": str(model_path) if model_path else None,
        "target_dataset": target_dataset,
    }

    if len(best_config_trials) == 1:
        best_trial = best_config_trials[0]
        return {
            **make_trial_payload(best_trial),
            **common_payload,
            "selection_strategy": "unique_best",
            "max_score_unique_configs": 1,
        }, best_trial

    cheapest_trial = select_cheapest_trial(best_config_trials)
    mode_trial = select_mode_trial(best_config_trials)
    first_best_trial = min(best_score_trials, key=lambda trial: trial.number)
    payload = {
        **make_trial_payload(first_best_trial),
        **common_payload,
        "selection_strategy": "tie_break_strategies",
        "max_score_unique_configs": len(best_config_trials),
        "selected_strategy": "first_best",
        "best_configs": {
            "first_best": make_trial_payload(first_best_trial),
            "cheapest": make_trial_payload(cheapest_trial),
            "mode": make_trial_payload(mode_trial),
        },
    }

    return payload, first_best_trial


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
    stale_top_params_path = output_dir / "top_10_parameters.json"
    if stale_top_params_path.exists():
        stale_top_params_path.unlink()

    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials.to_csv(trials_path, index=False)

    best_payload, best_trial = make_best_payload(
        study=study,
        objective_metrics=objective_metrics,
        metric_weights=metric_weights,
        metric_filters=metric_filters,
        model_path=model_path,
        target_dataset=target_dataset,
    )

    with best_params_path.open("w", encoding="utf-8") as result_file:
        json.dump(best_payload, result_file, indent=2, ensure_ascii=False)

    return trials_path, best_params_path, best_trial


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
        "--multi-objective",
        action="store_true",
        help=(
            "Optimize each metric in --metrics as a separate Optuna objective. "
            "best_parameters.json stores the Pareto front and selects knee-point "
            "representatives from it."
        ),
    )
    parser.add_argument(
        "--filter-best-metrics",
        action="store_true",
        help=(
            "Select the final best configuration only among trials whose predicted "
            "bert_score_recall, cosine_similarity, golden_doi_mrr, and "
            "rouge_l_recall meet the thresholds configured for --target-dataset."
        ),
    )
    parser.add_argument(
        "--filter-default-best-metrics",
        action="store_true",
        help=(
            "Select the final best configuration only among trials whose predicted "
            "bert_score_recall, cosine_similarity, golden_doi_mrr, and "
            "rouge_l_recall meet the default thresholds."
        ),
    )
    parser.add_argument(
        "--filter-top10-best-metrics",
        action="store_true",
        help=(
            "Select the final best configuration only among trials whose predicted "
            "bert_score_recall, cosine_similarity, golden_doi_mrr, and "
            "rouge_l_recall meet the top-10 thresholds configured for "
            "--target-dataset."
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
    filter_modes = [
        args.filter_best_metrics,
        args.filter_default_best_metrics,
        args.filter_top10_best_metrics,
    ]
    if sum(filter_modes) > 1:
        raise ValueError(
            "Use only one filter mode: --filter-best-metrics, "
            "--filter-default-best-metrics, or --filter-top10-best-metrics"
        )

    if args.filter_default_best_metrics:
        metric_filters = DEFAULT_BEST_METRIC_FILTERS
    elif args.filter_top10_best_metrics:
        metric_filters = resolve_top10_metric_filters(args.target_dataset)
    elif args.filter_best_metrics:
        metric_filters = resolve_best_metric_filters(args.target_dataset)
    else:
        metric_filters = None
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
        multi_objective=args.multi_objective,
    )
    metric_weights = normalize_weights(args.metrics, args.metric_weights)
    experiment_dir = resolve_experiment_dir(
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        model_path=args.model_path,
        target_dataset=args.target_dataset,
        metrics=args.metrics,
    )
    trials_path, best_params_path, best_trial = save_optimization_results(
        study=study,
        output_dir=experiment_dir,
        objective_metrics=args.metrics,
        metric_weights=metric_weights,
        metric_filters=metric_filters,
        model_path=args.model_path,
        target_dataset=args.target_dataset,
    )

    best_scalar_score = get_trial_scalar_score(
        best_trial,
        objective_metrics=args.metrics if args.multi_objective else None,
        metric_weights=metric_weights if args.multi_objective else None,
    )
    if args.multi_objective:
        print("Best representative weighted score:", f"{best_scalar_score:.6f}")
        print("Best representative objective values:")
        print(
            json.dumps(
                get_trial_objective_values(best_trial, objective_metrics=args.metrics),
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print("Best objective value:", f"{best_scalar_score:.6f}")
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
    print(
        "Optimization duration:",
        format_duration(getattr(study, "optimization_duration_seconds", 0.0)),
    )


if __name__ == "__main__":
    main()
