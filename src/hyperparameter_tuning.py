import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multioutput import MultiOutputRegressor

from train_models import (
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    evaluate_model,
    load_processed_data,
    to_numpy,
)


MODEL_SELECTION_METRICS_PATH = RESULTS_DIR / "model_selection_metrics.csv"
TUNED_MODEL_PATH = MODELS_DIR / "best_tuned_model.pkl"
TUNING_METRICS_PATH = RESULTS_DIR / "best_tuned_model_metrics.csv"
ALL_TUNED_MODELS_METRICS_PATH = RESULTS_DIR / "top_3_tuned_models_metrics.csv"
ALL_TUNED_TARGET_METRICS_PATH = RESULTS_DIR / "top_3_tuned_target_metrics.csv"
TUNING_METADATA_PATH = RESULTS_DIR / "best_tuned_model_metadata.json"


class OptunaSearchResult:
    """Small adapter with the same fields used from GridSearchCV."""

    def __init__(self, best_estimator, best_params, best_score, study):
        self.best_estimator_ = best_estimator
        self.best_params_ = best_params
        self.best_score_ = best_score
        self.study = study


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for top-3 baseline models."
    )
    parser.add_argument(
        "--method",
        choices=["grid", "optuna"],
        default="grid",
        help="Hyperparameter optimization method.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=40,
        help="Number of Optuna trials per model. Used only with --method optuna.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of top baseline models to tune.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help=(
            "Stop Optuna study after this many completed trials without improvement. "
            "Used only with --method optuna."
        ),
    )
    parser.add_argument(
        "--min-trials-before-stop",
        type=int,
        default=15,
        help=(
            "Minimum completed Optuna trials before patience-based stopping can run. "
            "Used only with --method optuna."
        ),
    )
    return parser.parse_args()


def load_top_model_names(metrics_path=MODEL_SELECTION_METRICS_PATH, top_n=3):
    """Read top model families selected by train_models.py."""
    metrics_path = Path(metrics_path)

    if not metrics_path.exists():
        raise FileNotFoundError(
            "Model selection metrics were not found. Run src/train_models.py first. "
            f"Missing: {metrics_path}"
        )

    metrics = pd.read_csv(metrics_path)
    required_columns = {"model", "rmse"}
    missing_columns = required_columns - set(metrics.columns)

    if missing_columns:
        raise ValueError(
            f"Model selection metrics must contain columns: {sorted(missing_columns)}"
        )

    return (
        metrics.sort_values("rmse", ascending=True)["model"]
        .head(top_n)
        .tolist()
    )


def build_estimator(model_name, random_state=RANDOM_STATE, params=None):
    """Build the selected model with stable base settings."""
    if model_name == "CatBoost":
        from catboost import CatBoostRegressor

        estimator = CatBoostRegressor(
            loss_function="MultiRMSE",
            random_seed=random_state,
            allow_writing_files=False,
            logging_level="Silent",
            thread_count=-1,
        )
    elif model_name == "XGBoost":
        from xgboost import XGBRegressor

        estimator = MultiOutputRegressor(
            XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=random_state,
                n_jobs=-1,
            )
        )
    elif model_name == "LightGBM":
        from lightgbm import LGBMRegressor

        estimator = MultiOutputRegressor(
            LGBMRegressor(
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            )
        )
    elif model_name == "HistGradientBoosting":
        estimator = MultiOutputRegressor(
            HistGradientBoostingRegressor(random_state=random_state)
        )
    elif model_name == "RandomForest":
        estimator = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"Unsupported model for tuning: {model_name}")

    if params:
        estimator.set_params(**params)

    return estimator


def get_param_grid(model_name):
    """Return a compact GridSearchCV hyperparameter grid."""
    param_grids = {
        "CatBoost": {
            "iterations": [250, 400],
            "learning_rate": [0.03, 0.06],
            "depth": [3, 4, 6],
            "l2_leaf_reg": [1, 3, 5],
        },
        "XGBoost": {
            "estimator__n_estimators": [250, 400],
            "estimator__learning_rate": [0.03, 0.06],
            "estimator__max_depth": [2, 3, 4],
            "estimator__subsample": [0.85, 1.0],
            "estimator__colsample_bytree": [0.85, 1.0],
        },
        "LightGBM": {
            "estimator__n_estimators": [250, 400],
            "estimator__learning_rate": [0.03, 0.06],
            "estimator__num_leaves": [7, 15, 31],
            "estimator__min_child_samples": [10, 20],
            "estimator__subsample": [0.85, 1.0],
        },
        "HistGradientBoosting": {
            "estimator__max_iter": [200, 300, 500],
            "estimator__learning_rate": [0.03, 0.06, 0.1],
            "estimator__max_leaf_nodes": [7, 15, 31],
            "estimator__min_samples_leaf": [10, 20],
            "estimator__l2_regularization": [0.0, 0.01, 0.1],
        },
        "RandomForest": {
            "n_estimators": [300, 500, 800],
            "max_depth": [None, 8, 16],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.75, 1.0],
        },
    }

    if model_name not in param_grids:
        raise ValueError(f"Unsupported model for tuning: {model_name}")

    return param_grids[model_name]


def suggest_optuna_params(trial, model_name):
    """Suggest hyperparameters for one Optuna trial."""
    if model_name == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 700, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }

    if model_name == "XGBoost":
        return {
            "estimator__n_estimators": trial.suggest_int(
                "estimator__n_estimators", 200, 700, step=100
            ),
            "estimator__learning_rate": trial.suggest_float(
                "estimator__learning_rate", 0.01, 0.12, log=True
            ),
            "estimator__max_depth": trial.suggest_int("estimator__max_depth", 2, 6),
            "estimator__subsample": trial.suggest_float(
                "estimator__subsample", 0.7, 1.0
            ),
            "estimator__colsample_bytree": trial.suggest_float(
                "estimator__colsample_bytree", 0.7, 1.0
            ),
        }

    if model_name == "LightGBM":
        return {
            "estimator__n_estimators": trial.suggest_int(
                "estimator__n_estimators", 200, 700, step=100
            ),
            "estimator__learning_rate": trial.suggest_float(
                "estimator__learning_rate", 0.01, 0.12, log=True
            ),
            "estimator__num_leaves": trial.suggest_int(
                "estimator__num_leaves", 7, 63
            ),
            "estimator__min_child_samples": trial.suggest_int(
                "estimator__min_child_samples", 5, 35
            ),
            "estimator__subsample": trial.suggest_float(
                "estimator__subsample", 0.7, 1.0
            ),
        }

    if model_name == "HistGradientBoosting":
        return {
            "estimator__max_iter": trial.suggest_int(
                "estimator__max_iter", 150, 700, step=50
            ),
            "estimator__learning_rate": trial.suggest_float(
                "estimator__learning_rate", 0.01, 0.12, log=True
            ),
            "estimator__max_leaf_nodes": trial.suggest_int(
                "estimator__max_leaf_nodes", 7, 63
            ),
            "estimator__min_samples_leaf": trial.suggest_int(
                "estimator__min_samples_leaf", 5, 35
            ),
            "estimator__l2_regularization": trial.suggest_float(
                "estimator__l2_regularization", 0.0, 0.2
            ),
        }

    if model_name == "RandomForest":
        max_depth = trial.suggest_categorical("max_depth", [None, 8, 12, 16, 24])
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=100),
            "max_depth": max_depth,
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", 0.75, 1.0]
            ),
        }

    raise ValueError(f"Unsupported model for tuning: {model_name}")


def tune_model_grid(estimator, param_grid, x_train, y_train):
    """Run cross-validated grid search and refit the best estimator."""
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(to_numpy(x_train), to_numpy(y_train))

    return search


def make_early_stopping_callback(patience, min_trials_before_stop):
    """Create an Optuna callback that stops after no improvement."""
    best_value = None
    best_trial_number = 0

    def callback(study, trial):
        nonlocal best_value, best_trial_number

        if trial.value is None:
            return

        if best_value is None or trial.value < best_value:
            best_value = trial.value
            best_trial_number = trial.number
            return

        completed_trials = len(
            [t for t in study.trials if t.state.name == "COMPLETE"]
        )
        no_improvement_rounds = trial.number - best_trial_number

        if (
            completed_trials >= min_trials_before_stop
            and no_improvement_rounds >= patience
        ):
            print(
                "Stopping Optuna early: "
                f"no improvement for {no_improvement_rounds} trials."
            )
            study.stop()

    return callback


def tune_model_optuna(
    model_name,
    x_train,
    y_train,
    n_trials,
    patience,
    min_trials_before_stop,
):
    """Run Optuna optimization and refit the best estimator."""
    import optuna

    x_train_values = to_numpy(x_train)
    y_train_values = to_numpy(y_train)
    cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = suggest_optuna_params(trial, model_name)
        fold_scores = []

        for train_index, valid_index in cv.split(x_train_values):
            estimator = build_estimator(model_name, params=params)
            estimator.fit(x_train_values[train_index], y_train_values[train_index])
            predictions = estimator.predict(x_train_values[valid_index])
            rmse = np.sqrt(
                mean_squared_error(y_train_values[valid_index], predictions)
            )
            fold_scores.append(rmse)
            trial.report(float(np.mean(fold_scores)), step=len(fold_scores))

            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min_trials_before_stop,
        n_warmup_steps=1,
    )
    early_stopping_callback = make_early_stopping_callback(
        patience=patience,
        min_trials_before_stop=min_trials_before_stop,
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[early_stopping_callback],
    )

    best_estimator = build_estimator(model_name, params=study.best_params)
    best_estimator.fit(x_train_values, y_train_values)

    return OptunaSearchResult(
        best_estimator=best_estimator,
        best_params=study.best_params,
        best_score=-float(study.best_value),
        study=study,
    )


def tune_model(
    model_name,
    method,
    x_train,
    y_train,
    n_trials,
    patience,
    min_trials_before_stop,
):
    """Tune one model with GridSearchCV or Optuna."""
    if method == "grid":
        estimator = build_estimator(model_name)
        param_grid = get_param_grid(model_name)
        return tune_model_grid(estimator, param_grid, x_train, y_train)

    if method == "optuna":
        return tune_model_optuna(
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            n_trials=n_trials,
            patience=patience,
            min_trials_before_stop=min_trials_before_stop,
        )

    raise ValueError(f"Unsupported tuning method: {method}")


def metrics_by_target(metrics):
    """Create a dataframe with RMSE, MAE, and R2 for every target."""
    rows = []

    for target_name, rmse_value in metrics["rmse_per_target"].items():
        rows.append(
            {
                "target": target_name,
                "rmse": rmse_value,
                "mae": metrics["mae_per_target"][target_name],
                "r2": metrics["r2_per_target"][target_name],
            }
        )

    return pd.DataFrame(rows)


def save_tuning_results(best_result, summary_df, target_metrics_df, method):
    """Save best tuned model and metrics for all tuned models."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with TUNED_MODEL_PATH.open("wb") as model_file:
        pickle.dump(best_result["search"].best_estimator_, model_file)

    best_target_metrics = target_metrics_df[
        target_metrics_df["model"] == best_result["model"]
    ].drop(columns=["model"])

    best_target_metrics.to_csv(TUNING_METRICS_PATH, index=False)
    summary_df.to_csv(ALL_TUNED_MODELS_METRICS_PATH, index=False)
    target_metrics_df.to_csv(ALL_TUNED_TARGET_METRICS_PATH, index=False)

    metadata = {
        "method": method,
        "optuna_early_stopping": (
            "MedianPruner plus patience callback" if method == "optuna" else None
        ),
        "model": best_result["model"],
        "tuned_models": summary_df["model"].tolist(),
        "selection_metric": "lowest tuned test RMSE among top baseline models",
        "best_params": best_result["search"].best_params_,
        "best_cv_rmse": best_result["best_cv_rmse"],
        "test_metrics": {
            "rmse": best_result["metrics"]["rmse"],
            "mae": best_result["metrics"]["mae"],
            "r2": best_result["metrics"]["r2"],
        },
        "tuned_model_path": str(TUNED_MODEL_PATH),
        "metrics_path": str(TUNING_METRICS_PATH),
        "all_tuned_models_metrics_path": str(ALL_TUNED_MODELS_METRICS_PATH),
        "all_tuned_target_metrics_path": str(ALL_TUNED_TARGET_METRICS_PATH),
    }
    TUNING_METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    top_model_names = load_top_model_names(top_n=args.top_n)
    x_train, x_test, y_train, y_test = load_processed_data()

    tuning_results = []
    target_metrics_frames = []

    for model_name in top_model_names:
        print(f"\nTuning model: {model_name} with {args.method}")
        search = tune_model(
            model_name=model_name,
            method=args.method,
            x_train=x_train,
            y_train=y_train,
            n_trials=args.n_trials,
            patience=args.patience,
            min_trials_before_stop=args.min_trials_before_stop,
        )
        metrics = evaluate_model(search.best_estimator_, x_test, y_test)
        target_metrics = metrics_by_target(metrics)
        target_metrics.insert(0, "model", model_name)
        target_metrics_frames.append(target_metrics)

        tuning_results.append(
            {
                "model": model_name,
                "search": search,
                "metrics": metrics,
                "best_cv_rmse": float(-search.best_score_),
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "best_params": search.best_params_,
            }
        )

    best_result = min(tuning_results, key=lambda result: result["rmse"])
    summary_df = pd.DataFrame(
        [
            {
                "method": args.method,
                "model": result["model"],
                "best_cv_rmse": result["best_cv_rmse"],
                "test_rmse": result["rmse"],
                "test_mae": result["mae"],
                "test_r2": result["r2"],
                "best_params": json.dumps(result["best_params"]),
            }
            for result in tuning_results
        ]
    ).sort_values("test_rmse", ascending=True)
    target_metrics = pd.concat(target_metrics_frames, ignore_index=True)

    save_tuning_results(best_result, summary_df, target_metrics, args.method)

    print("\nTuned model summary:")
    print(summary_df.to_string(index=False))
    print("\nBest tuned model:")
    print(best_result["model"])
    print("\nBest parameters:")
    print(json.dumps(best_result["search"].best_params_, indent=2))
    print(f"\nBest CV RMSE: {best_result['best_cv_rmse']:.6f}")
    print("\nBest tuned model test metrics by target:")
    print(
        target_metrics[target_metrics["model"] == best_result["model"]]
        .drop(columns=["model"])
        .to_string(index=False)
    )
    print(
        "\nMean test metrics: "
        f"RMSE={best_result['metrics']['rmse']:.6f}, "
        f"MAE={best_result['metrics']['mae']:.6f}, "
        f"R2={best_result['metrics']['r2']:.6f}"
    )
    print(f"\nSaved tuned model to {TUNED_MODEL_PATH}")
    print(f"Saved target metrics to {TUNING_METRICS_PATH}")
    print(f"Saved all tuned model metrics to {ALL_TUNED_MODELS_METRICS_PATH}")
    print(f"Saved all tuned target metrics to {ALL_TUNED_TARGET_METRICS_PATH}")
    print(f"Saved metadata to {TUNING_METADATA_PATH}")


if __name__ == "__main__":
    main()
