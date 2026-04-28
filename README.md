# RAG System Design ml Project

This project involves training ML models to predict RAG metrics based on parameters.

## Steps:
1. Data preprocessing from input table (e.g., oxazo_results.csv)
2. Testing 5 different boosting models from scikit-learn
3. Hyperparameter tuning for 3 best models
4. Best model testing and metrics collection

## Project Structure:
- data/: Input and processed data files
- src/: Source code scripts
- results/: Evaluation results and metrics
- notebooks/: Jupyter notebooks for EDA

## Running ML Scripts

Run commands from the project root through Poetry:

```bash
poetry install
poetry run python --version
```

Use `poetry run python ...` for all scripts so they run inside the project virtual
environment with the dependencies from `pyproject.toml`.

### 1. Preprocess data

Preprocess one raw results table:

```bash
poetry run python src/ml/preprocess.py data/rag_results/oxazo_results.csv
```

The script writes processed files to `data/processed/<dataset_name>/train.csv` and
`data/processed/<dataset_name>/test.csv`.

### 2. Train baseline models

Train and evaluate baseline models on one processed dataset:

```bash
poetry run python src/ml/train_models.py data/processed/oxazo
```

You can also pass several processed datasets. In this case the script combines them:

```bash
poetry run python src/ml/train_models.py data/processed/complexes data/processed/oxazo
```

For explicit transfer experiments, pass source train dataset(s) and target test
dataset(s) separately:

```bash
poetry run python src/ml/train_models.py --train-data-dir data/processed/complexes --test-data-dir data/processed/nanozymes
```

Outputs are saved to `results/<dataset_name>/train_models/model_selection_metrics.csv`.

### 3. Hyperparameter tuning

Tune the top baseline models selected by `train_models.py`:

```bash
poetry run python src/ml/hyperparameter_tuning.py data/processed/oxazo
```

By default, tuning uses grid search for the top 3 models. You can change the method to Optuna,
number of top models, and the number of Optuna trials:

```bash
poetry run python src/ml/hyperparameter_tuning.py --method optuna --top-n 3 --n-trials 40 data/processed/oxazo
```

For explicit transfer experiments, use the same source/target split as in
`train_models.py`:

```bash
poetry run python src/ml/hyperparameter_tuning.py --train-data-dir data/processed/complexes --test-data-dir data/processed/nanozymes
```

Outputs are saved to `results/<dataset_name>/hp_tuning/`, including:

- `best_tuned_model.pkl`
- `top_3_tuned_models_metrics.csv`
- `top_3_tuned_target_metrics.csv`

### 4. SHAP feature importance

Calculate SHAP values for each target metric and for the mean across all targets:

```bash
poetry run python src/ml/shap_feature_importance.py -m results/oxazo/hp_tuning/best_tuned_model.pkl -d data/processed/oxazo
```

Outputs are saved to `results/<dataset_name>/shap_feature_importance/`.

### 5. shapiq feature interactions

Calculate second-order feature interactions for the mean across all targets:

```bash
poetry run python src/ml/shapiq_feature_interactions.py -m results/oxazo/hp_tuning/best_tuned_model.pkl -d data/processed/oxazo
```

Outputs are saved to `results/<dataset_name>/shapiq_feature_interactions/`.
