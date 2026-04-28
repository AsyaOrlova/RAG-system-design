import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multioutput import MultiOutputRegressor

sys.path.append(str(Path(__file__).resolve().parents[1]))

from ml.train_models import (
    build_model_pipeline,
    evaluate_model,
    get_feature_types,
    get_output_dir,
    get_train_models_dir,
    get_transfer_output_dir,
    get_transfer_train_models_dir,
    load_processed_data,
    load_transfer_data,
)

from ml.constants import TRAIN_MODELS_RANDOM_STATE

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
        "-r",
        "--results-dir",
        type=Path,
        help=(
            "Path to a results directory with model_selection_metrics.csv, "
            "or directly to model_selection_metrics.csv."
        ),
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        dest="data_dir_options",
        action="append",
        nargs="+",
        type=Path,
        help=(
            "Path(s) to processed train/test data. Can be used multiple times. "
        ),
    )
    parser.add_argument(
        "--train-data-dir",
        dest="train_data_dir_options",
        action="append",
        nargs="+",
        type=Path,
        help="Source processed dataset directory/directories used for training.",
    )
    parser.add_argument(
        "--test-data-dir",
        dest="test_data_dir_options",
        action="append",
        nargs="+",
        type=Path,
        help="Target processed dataset directory/directories used for testing.",
    )
    parser.add_argument(
        "data_dirs",
        nargs="*",
        type=Path,
        help="Path(s) to directories with processed train.csv and test.csv files.",
    )
    args = parser.parse_args()
    option_dirs = [
        data_dir
        for data_dir_group in (args.data_dir_options or [])
        for data_dir in data_dir_group
    ]
    args.data_dirs = option_dirs or args.data_dirs
    args.train_data_dirs = [
        data_dir
        for data_dir_group in (args.train_data_dir_options or [])
        for data_dir in data_dir_group
    ]
    args.test_data_dirs = [
        data_dir
        for data_dir_group in (args.test_data_dir_options or [])
        for data_dir in data_dir_group
    ]
    if args.data_dirs and str(args.data_dirs[0]) == "data_dirs":
        args.data_dirs = args.data_dirs[1:]
    del args.data_dir_options
    del args.train_data_dir_options
    del args.test_data_dir_options
    if bool(args.train_data_dirs) != bool(args.test_data_dirs):
        parser.error("--train-data-dir and --test-data-dir must be used together")
    return args


def get_metrics_path(results_path):
    """Resolve model-selection metrics path from a results directory or CSV file."""
    results_path = Path(results_path)
    if results_path.suffix.lower() == ".csv":
        return results_path

    if results_path.name == "train_models":
        return results_path / "model_selection_metrics.csv"

    return results_path / "train_models" / "model_selection_metrics.csv"


def get_results_dir(results_path):
    """Resolve output directory from a results directory or CSV file."""
    results_path = Path(results_path)
    if results_path.suffix.lower() == ".csv":
        parent = results_path.parent
        if parent.name == "train_models":
            return parent.parent / "hp_tuning"
        return parent

    if results_path.name in {"train_models", "hp_tuning"}:
        return results_path.parent / "hp_tuning"

    return results_path / "hp_tuning"


def load_top_model_names(metrics_path, top_n=3):
    """Read top model families selected by train_models.py."""
    metrics_path = Path(metrics_path)

    if not metrics_path.exists():
        raise FileNotFoundError(
            "Model selection metrics were not found. Run src/train_models.py first. "
            f"Missing: {metrics_path}"
        )

    metrics = pd.read_csv(metrics_path)
    required_columns = {"model", "cv_rmse"}
    missing_columns = required_columns - set(metrics.columns)

    if missing_columns:
        raise ValueError(
            f"Model selection metrics must contain columns: {sorted(missing_columns)}"
        )

    return (
        metrics.sort_values("cv_rmse", ascending=True)["model"]
        .head(top_n)
        .tolist()
    )


def build_estimator(
    model_name,
    numeric_features,
    categorical_features,
    random_state=TRAIN_MODELS_RANDOM_STATE,
    params=None,
):
    """Build the selected model with stable base settings."""
    if model_name == "CatBoost":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            loss_function="MultiRMSE",
            random_seed=random_state,
            allow_writing_files=False,
            logging_level="Silent",
            thread_count=-1,
        )
    elif model_name == "XGBoost":
        from xgboost import XGBRegressor

        model = MultiOutputRegressor(
            XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                random_state=random_state,
                n_jobs=-1,
            )
        )
    elif model_name == "LightGBM":
        from lightgbm import LGBMRegressor

        model = MultiOutputRegressor(
            LGBMRegressor(
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            )
        )
    elif model_name == "ExtraTrees":
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor(random_state=random_state, n_jobs=-1)
    elif model_name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    else:
        raise ValueError(f"Unsupported model for tuning: {model_name}")

    estimator = build_model_pipeline(
        model,
        numeric_features,
        categorical_features,
    )

    if params:
        estimator.set_params(**params)

    return estimator


def get_param_grid(model_name):
    """Return a compact GridSearchCV hyperparameter grid."""
    param_grids = {
        "CatBoost": {
            "model__iterations": [250, 400],
            "model__learning_rate": [0.03, 0.06],
            "model__depth": [3, 4, 6],
            "model__l2_leaf_reg": [1, 3, 5],
        },
        "XGBoost": {
            "model__estimator__n_estimators": [250, 400],
            "model__estimator__learning_rate": [0.03, 0.06],
            "model__estimator__max_depth": [2, 3, 4],
            "model__estimator__subsample": [0.85, 1.0],
            "model__estimator__colsample_bytree": [0.85, 1.0],
        },
        "LightGBM": {
            "model__estimator__n_estimators": [250, 400],
            "model__estimator__learning_rate": [0.03, 0.06],
            "model__estimator__num_leaves": [7, 15, 31],
            "model__estimator__min_child_samples": [10, 20],
            "model__estimator__subsample": [0.85, 1.0],
        },
        "ExtraTrees": {
            "model__n_estimators": [300, 500, 800],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.75, 1.0],
        },
        "RandomForest": {
            "model__n_estimators": [300, 500, 800],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.75, 1.0],
        },
    }

    if model_name not in param_grids:
        raise ValueError(f"Unsupported model for tuning: {model_name}")

    return param_grids[model_name]


def suggest_optuna_params(trial, model_name):
    """Suggest hyperparameters for one Optuna trial."""
    if model_name == "CatBoost":
        return {
            "model__iterations": trial.suggest_int("model__iterations", 200, 700, step=100),
            "model__learning_rate": trial.suggest_float("model__learning_rate", 0.01, 0.12, log=True),
            "model__depth": trial.suggest_int("model__depth", 3, 8),
            "model__l2_leaf_reg": trial.suggest_float("model__l2_leaf_reg", 1.0, 10.0),
        }

    if model_name == "XGBoost":
        return {
            "model__estimator__n_estimators": trial.suggest_int(
                "model__estimator__n_estimators", 200, 700, step=100
            ),
            "model__estimator__learning_rate": trial.suggest_float(
                "model__estimator__learning_rate", 0.01, 0.12, log=True
            ),
            "model__estimator__max_depth": trial.suggest_int("model__estimator__max_depth", 2, 6),
            "model__estimator__subsample": trial.suggest_float(
                "model__estimator__subsample", 0.7, 1.0
            ),
            "model__estimator__colsample_bytree": trial.suggest_float(
                "model__estimator__colsample_bytree", 0.7, 1.0
            ),
        }

    if model_name == "LightGBM":
        return {
            "model__estimator__n_estimators": trial.suggest_int(
                "model__estimator__n_estimators", 200, 700, step=100
            ),
            "model__estimator__learning_rate": trial.suggest_float(
                "model__estimator__learning_rate", 0.01, 0.12, log=True
            ),
            "model__estimator__num_leaves": trial.suggest_int(
                "model__estimator__num_leaves", 7, 63
            ),
            "model__estimator__min_child_samples": trial.suggest_int(
                "model__estimator__min_child_samples", 5, 35
            ),
            "model__estimator__subsample": trial.suggest_float(
                "model__estimator__subsample", 0.7, 1.0
            ),
        }

    if model_name == "ExtraTrees":
        max_depth = trial.suggest_categorical("model__max_depth", [None, 8, 12, 16, 24])
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 300, 900, step=100),
            "model__max_depth": max_depth,
            "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 5),
            "model__max_features": trial.suggest_categorical(
                "model__max_features", ["sqrt", 0.75, 1.0]
            ),
        }

    if model_name == "RandomForest":
        max_depth = trial.suggest_categorical("model__max_depth", [None, 8, 12, 16, 24])
        return {
            "model__n_estimators": trial.suggest_int("model__n_estimators", 300, 900, step=100),
            "model__max_depth": max_depth,
            "model__min_samples_leaf": trial.suggest_int("model__min_samples_leaf", 1, 5),
            "model__max_features": trial.suggest_categorical(
                "model__max_features", ["sqrt", 0.75, 1.0]
            ),
        }

    raise ValueError(f"Unsupported model for tuning: {model_name}")


def tune_model_grid(estimator, param_grid, x_train, y_train):
    """Run cross-validated grid search and refit the best estimator."""
    cv = KFold(n_splits=3, shuffle=True, random_state=TRAIN_MODELS_RANDOM_STATE)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(x_train, y_train)

    return search


def tune_model_optuna(
    model_name,
    x_train,
    y_train,
    numeric_features,
    categorical_features,
    n_trials
):
    """Run Optuna optimization and refit the best estimator."""
    import optuna

    cv = KFold(n_splits=3, shuffle=True, random_state=TRAIN_MODELS_RANDOM_STATE)

    def objective(trial):
        params = suggest_optuna_params(trial, model_name)
        fold_scores = []

        for train_index, valid_index in cv.split(x_train):
            estimator = build_estimator(
                model_name,
                numeric_features,
                categorical_features,
                params=params,
            )
            x_train_fold = x_train.iloc[train_index]
            y_train_fold = y_train.iloc[train_index]
            x_valid_fold = x_train.iloc[valid_index]
            y_valid_fold = y_train.iloc[valid_index]

            estimator.fit(x_train_fold, y_train_fold)
            predictions = estimator.predict(x_valid_fold)
            rmse = np.sqrt(
                mean_squared_error(y_valid_fold, predictions)
            )
            fold_scores.append(rmse)
            trial.report(float(np.mean(fold_scores)), step=len(fold_scores))

            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=TRAIN_MODELS_RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(
        n_warmup_steps=1,
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        objective,
        n_trials=n_trials,
    )

    best_estimator = build_estimator(
        model_name,
        numeric_features,
        categorical_features,
        params=study.best_params,
    )
    best_estimator.fit(x_train, y_train)

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
    numeric_features,
    categorical_features,
    n_trials,
):
    """Tune one model with GridSearchCV or Optuna."""
    if method == "grid":
        estimator = build_estimator(
            model_name,
            numeric_features,
            categorical_features,
        )
        param_grid = get_param_grid(model_name)
        return  (estimator, param_grid, x_train, y_train)

    if method == "optuna":
        return tune_model_optuna(
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            n_trials=n_trials
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


def save_tuning_results(
    best_result,
    summary_df,
    target_metrics_df,
    output_dir,
    preprocessing_metadata=None,
):
    """Save tuned model and metrics for all tuned models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tuned_model_path = output_dir / "best_tuned_model.pkl"
    model_bundle = {
        "model": best_result["search"].best_estimator_,
        "preprocessing": preprocessing_metadata or {},
    }
    with tuned_model_path.open("wb") as model_file:
        pickle.dump(model_bundle, model_file)

    all_models_metrics_path = output_dir / "top_3_tuned_models_metrics.csv"
    all_target_metrics_path = output_dir / "top_3_tuned_target_metrics.csv"
    summary_df.to_csv(all_models_metrics_path, index=False)
    target_metrics_df.to_csv(all_target_metrics_path, index=False)

    return tuned_model_path, all_models_metrics_path, all_target_metrics_path


def main():
    args = parse_args()
    is_transfer = bool(args.train_data_dirs and args.test_data_dirs)
    output_dir = (
        get_results_dir(args.results_dir)
        if args.results_dir
        else get_transfer_output_dir(args.train_data_dirs, args.test_data_dirs) / "hp_tuning"
        if is_transfer
        else get_output_dir(args.data_dirs) / "hp_tuning"
    )
    metrics_path = (
        get_metrics_path(args.results_dir)
        if args.results_dir
        else get_metrics_path(
            get_transfer_train_models_dir(args.train_data_dirs, args.test_data_dirs)
        )
        if is_transfer
        else get_metrics_path(get_train_models_dir(args.data_dirs))
    )
    top_model_names = load_top_model_names(metrics_path=metrics_path, top_n=args.top_n)
    if is_transfer:
        (
            x_train,
            x_test,
            y_train,
            y_test,
            preprocessing_metadata,
        ) = load_transfer_data(
            args.train_data_dirs,
            args.test_data_dirs,
            return_metadata=True,
        )
        loaded_data_message = (
            "Loaded train data from "
            f"{', '.join(str(path) for path in args.train_data_dirs)} "
            "and test data from "
            f"{', '.join(str(path) for path in args.test_data_dirs)}"
        )
    else:
        (
            x_train,
            x_test,
            y_train,
            y_test,
            preprocessing_metadata,
        ) = load_processed_data(args.data_dirs, return_metadata=True)
        loaded_data_message = (
            "Loaded processed data from "
            f"{', '.join(str(path) for path in args.data_dirs)}"
        )
    numeric_features, categorical_features = get_feature_types(x_train)

    tuning_results = []
    target_metrics_frames = []

    for model_name in top_model_names:
        print(f"\nTuning model: {model_name} with {args.method}")
        search = tune_model(
            model_name=model_name,
            method=args.method,
            x_train=x_train,
            y_train=y_train,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            n_trials=args.n_trials
        )
        train_metrics = evaluate_model(search.best_estimator_, x_train, y_train)
        test_metrics = evaluate_model(search.best_estimator_, x_test, y_test)

        train_target_metrics = metrics_by_target(train_metrics)
        train_target_metrics.insert(0, "split", "train")
        train_target_metrics.insert(0, "model", model_name)

        test_target_metrics = metrics_by_target(test_metrics)
        test_target_metrics.insert(0, "split", "test")
        test_target_metrics.insert(0, "model", model_name)

        target_metrics_frames.extend([train_target_metrics, test_target_metrics])

        tuning_results.append(
            {
                "model": model_name,
                "search": search,
                "metrics": test_metrics,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "best_cv_rmse": float(-search.best_score_),
                "rmse": test_metrics["rmse"],
                "mae": test_metrics["mae"],
                "r2": test_metrics["r2"],
                "best_params": search.best_params_,
            }
        )

    best_result = min(tuning_results, key=lambda result: result["best_cv_rmse"])
    summary_df = pd.DataFrame(
        [
            {
                "method": args.method,
                "model": result["model"],
                "best_cv_rmse": result["best_cv_rmse"],
                "train_rmse": result["train_metrics"]["rmse"],
                "train_mae": result["train_metrics"]["mae"],
                "train_r2": result["train_metrics"]["r2"],
                "test_rmse": result["rmse"],
                "test_mae": result["mae"],
                "test_r2": result["r2"],
                "best_params": json.dumps(result["best_params"]),
            }
            for result in tuning_results
        ]
    ).sort_values("best_cv_rmse", ascending=True)
    target_metrics = pd.concat(target_metrics_frames, ignore_index=True)

    (
        tuned_model_path,
        all_models_metrics_path,
        all_target_metrics_path,
    ) = save_tuning_results(
        best_result,
        summary_df,
        target_metrics,
        output_dir,
        preprocessing_metadata,
    )

    print("\nTuned model summary:")
    print(summary_df.to_string(index=False))
    print("\nBest tuned model:")
    print(best_result["model"])
    print("\nBest parameters:")
    print(json.dumps(best_result["search"].best_params_, indent=2))
    print(f"\nBest CV RMSE: {best_result['best_cv_rmse']:.6f}")
    print("\nBest tuned model train/test metrics by target:")
    print(
        target_metrics[target_metrics["model"] == best_result["model"]]
        .drop(columns=["model"])
        .to_string(index=False)
    )
    print(
        "\nMean train metrics: "
        f"RMSE={best_result['train_metrics']['rmse']:.6f}, "
        f"MAE={best_result['train_metrics']['mae']:.6f}, "
        f"R2={best_result['train_metrics']['r2']:.6f}"
    )
    print(
        "Mean test metrics: "
        f"RMSE={best_result['test_metrics']['rmse']:.6f}, "
        f"MAE={best_result['test_metrics']['mae']:.6f}, "
        f"R2={best_result['test_metrics']['r2']:.6f}"
    )
    print(f"\nSaved tuned model to {tuned_model_path}")
    print(f"Loaded model selection metrics from {metrics_path}")
    print(loaded_data_message)
    print(f"Saved all tuned model metrics to {all_models_metrics_path}")
    print(f"Saved all tuned target metrics to {all_target_metrics_path}")


if __name__ == "__main__":
    main()
