import argparse
import copy
import pickle
from pathlib import Path

import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

from test_saved_model import (
    add_required_summary_features,
    align_features,
    apply_summary_standardization,
    get_expected_feature_names,
    load_model,
    load_train_test_data,
    metrics_by_target,
)
from ML.train_models import PROCESSED_DATA_DIR, RESULTS_DIR, evaluate_model, split_features_targets


DEFAULT_MODEL_PATH = RESULTS_DIR / PROCESSED_DATA_DIR.name / "hp_tuning" / "best_tuned_model.pkl"


def get_finetune_dir(model_path):
    """Resolve finetune output directory inside the dataset results folder."""
    model_dir = Path(model_path).parent
    if model_dir.name in {"train_models", "hp_tuning", "finetune"}:
        return model_dir.parent / "finetune"

    return model_dir / "finetune"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue training a saved model on part of a processed dataset and test on the remaining part."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        type=Path,
        help="Path to a processed dataset directory with train.csv and test.csv.",
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        dest="data_dir_option",
        type=Path,
        help=f"Path to processed train/test data. Defaults to {PROCESSED_DATA_DIR}",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to saved .pkl model. Defaults to {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--finetune-size",
        type=float,
        default=0.8,
        help="Fraction of the dataset to use for continued training.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for the finetune/test split.",
    )
    parser.add_argument(
        "--extra-iterations",
        type=int,
        default=100,
        help="Additional boosting iterations for supported boosting models.",
    )
    parser.add_argument(
        "--extra-estimators",
        type=int,
        default=100,
        help="Additional trees for random-forest style warm-start training.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Directory to save the finetuned model and metrics. Defaults to results/<dataset>/finetune.",
    )

    args = parser.parse_args()
    args.data_dir = args.data_dir_option or args.data_dir or PROCESSED_DATA_DIR
    args.output_dir = args.output_dir or get_finetune_dir(args.model_path)
    if not 0 < args.finetune_size < 1:
        parser.error("--finetune-size must be between 0 and 1")

    del args.data_dir_option
    return args


def split_finetune_test(features, targets, finetune_size=0.8, random_state=42):
    shuffled_index = features.sample(frac=1, random_state=random_state).index
    finetune_count = max(1, round(len(shuffled_index) * finetune_size))
    if finetune_count >= len(shuffled_index):
        finetune_count = len(shuffled_index) - 1

    finetune_index = shuffled_index[:finetune_count]
    test_index = shuffled_index[finetune_count:]

    return (
        features.loc[finetune_index].copy(),
        features.loc[test_index].copy(),
        targets.loc[finetune_index].copy(),
        targets.loc[test_index].copy(),
    )


def continue_fit_single_target_model(model, x_train, y_train, extra_iterations, extra_estimators):
    model = copy.deepcopy(model)
    model_name = model.__class__.__name__

    if model_name == "CatBoostRegressor":
        params = model.get_params()
        current_iterations = params.get("iterations") or model.tree_count_
        model.set_params(
            iterations=current_iterations + extra_iterations,
            allow_writing_files=False,
            logging_level="Silent",
        )
        model.fit(x_train, y_train, init_model=model)
        return model

    if model_name == "XGBRegressor":
        params = model.get_params()
        current_estimators = params.get("n_estimators", 0)
        model.set_params(n_estimators=current_estimators + extra_iterations)
        model.fit(x_train, y_train, xgb_model=model.get_booster())
        return model

    if model_name == "LGBMRegressor":
        params = model.get_params()
        current_estimators = params.get("n_estimators", 0)
        model.set_params(n_estimators=current_estimators + extra_iterations)
        model.fit(x_train, y_train, init_model=model.booster_)
        return model

    if model_name == "HistGradientBoostingRegressor":
        current_iter = model.get_params().get("max_iter", 0)
        model.set_params(warm_start=True, max_iter=current_iter + extra_iterations)
        model.fit(x_train, y_train)
        return model

    if model_name == "RandomForestRegressor":
        current_estimators = model.get_params().get("n_estimators", 0)
        model.set_params(
            warm_start=True,
            n_estimators=current_estimators + extra_estimators,
        )
        model.fit(x_train, y_train)
        return model

    raise ValueError(f"Continued training is not implemented for {model_name}")


def continue_fit_model(model, x_train, y_train, extra_iterations, extra_estimators):
    if isinstance(model, MultiOutputRegressor):
        adapted_model = copy.deepcopy(model)
        adapted_estimators = []

        for index, estimator in enumerate(model.estimators_):
            adapted_estimators.append(
                continue_fit_single_target_model(
                    estimator,
                    x_train,
                    y_train.iloc[:, index],
                    extra_iterations,
                    extra_estimators,
                )
            )

        adapted_model.estimators_ = adapted_estimators
        if hasattr(model, "n_features_in_"):
            adapted_model.n_features_in_ = model.n_features_in_
        if hasattr(model, "feature_names_in_"):
            adapted_model.feature_names_in_ = model.feature_names_in_
        return adapted_model

    return continue_fit_single_target_model(
        model,
        x_train,
        y_train,
        extra_iterations,
        extra_estimators,
    )


def save_outputs(
    finetuned_model,
    preprocessing,
    metrics,
    target_metrics,
    output_dir,
    model_path,
    dataset_name,
    finetune_size,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    finetune_suffix = str(finetune_size).rstrip("0").rstrip(".")
    finetuned_model_path = (
        output_dir
        / f"{Path(model_path).stem}_finetuned_{finetune_suffix}_on_{dataset_name}.pkl"
    )
    summary_path = (
        output_dir
        / f"finetuned_model_test_{finetune_suffix}_{dataset_name}_metrics.csv"
    )
    target_metrics_path = (
        output_dir
        / f"finetuned_model_test_{finetune_suffix}_{dataset_name}_target_metrics.csv"
    )

    saved_object = (
        {"model": finetuned_model, "preprocessing": preprocessing}
        if preprocessing
        else finetuned_model
    )

    with finetuned_model_path.open("wb") as model_file:
        pickle.dump(saved_object, model_file)

    pd.DataFrame(
        [{"rmse": metrics["rmse"], "mae": metrics["mae"], "r2": metrics["r2"]}]
    ).to_csv(summary_path, index=False)
    target_metrics.to_csv(target_metrics_path, index=False)

    return finetuned_model_path, summary_path, target_metrics_path


def main():
    args = parse_args()
    model, preprocessing = load_model(args.model_path)
    data = load_train_test_data(args.data_dir)
    features, targets = split_features_targets(data)
    features = add_required_summary_features(features, args.data_dir, preprocessing)
    expected_feature_names = get_expected_feature_names(model, preprocessing)
    features = align_features(features, expected_feature_names)
    features = apply_summary_standardization(features, preprocessing)

    x_finetune, x_test, y_finetune, y_test = split_finetune_test(
        features,
        targets,
        finetune_size=args.finetune_size,
        random_state=args.random_state,
    )
    finetuned_model = continue_fit_model(
        model,
        x_finetune,
        y_finetune,
        args.extra_iterations,
        args.extra_estimators,
    )

    metrics = evaluate_model(finetuned_model, x_test, y_test)
    target_metrics = metrics_by_target(metrics)
    dataset_name = Path(args.data_dir).name
    finetuned_model_path, summary_path, target_metrics_path = save_outputs(
        finetuned_model,
        preprocessing,
        metrics,
        target_metrics,
        args.output_dir,
        args.model_path,
        dataset_name,
        args.finetune_size,
    )

    print("Finetuned model test metrics:")
    print(f"RMSE={metrics['rmse']:.6f}, MAE={metrics['mae']:.6f}, R2={metrics['r2']:.6f}")
    print(f"Loaded base model from {args.model_path}")
    print(f"Loaded processed data from {args.data_dir}")
    print(f"Saved finetuned model to {finetuned_model_path}")
    print(f"Saved summary metrics to {summary_path}")
    print(f"Saved target metrics to {target_metrics_path}")


if __name__ == "__main__":
    main()
