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

By default, preprocessing adds dataset-level summary features from
`data/db/summary.csv`. To skip these features, pass `--no-summary-features`:

```bash
poetry run python src/ml/preprocess.py data/rag_results/oxazo_results.csv --no-summary-features
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

### 2.1. LightGBM dataset mix sweep

Train LightGBM models on the full first dataset train split plus a varying
share of the second dataset train split, from +0% to +100% in 10% steps, and
plot target-level R2 on the second dataset test split:

```bash
poetry run python src/ml/run_lightgbm_dataset_mix_sweep.py data/processed/complexes data/processed/oxazo
```

Outputs are saved to
`results/<first>_plus_<second>_lightgbm_sweep/`: trained model `.pkl` files,
summary metrics, target metrics, and `r2_by_dataset_mix.png`.
For each point, the first dataset is used fully and the second dataset share is
sampled from its `train.csv`. Evaluation always uses only the second dataset's
full `test.csv`.
The plot includes R2 lines for `bert_score_recall`, `cosine_similarity`,
`golden_doi_mrr`, `golden_doi_recall`, `rouge_l_recall`, and
`scientific_fact_recall`.

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

### 4.1. Finetune size sweep

Finetune a saved model with train sizes from 0.1 to 0.9 and draw the mean R2 plot:

```bash
poetry run python src/ml/run_finetune_size_sweep.py data/processed/complexes -m results/complexes/hp_tuning/best_tuned_model.pkl
```

The script runs `src/ml/finetune_saved_model.py` for each `--finetune-size` value and
then calls `src/viz/plot_finetune_r2.py`. Outputs are saved to
`results/<dataset_name>/finetune/` by default.

### 5. shapiq feature interactions

Calculate second-order feature interactions for the mean across all targets:

```bash
poetry run python src/feature_importance/shapiq_feature_interactions.py -m results/oxazo/hp_tuning/best_tuned_model.pkl -d data/processed/oxazo
```

Outputs are saved to `results/<dataset_name>/shapiq_feature_interactions/`.

Redraw a graph from already saved shapiq results without recomputing interactions:

```bash
poetry run python src/feature_importance/shapiq_feature_interactions.py \
  --plot-from-results results/oxazo/shapiq_feature_interactions/feature_interactions_all_metrics.csv
```

The script infers the matching
`results/<dataset_name>/shap_feature_importance/shap_values_<target>.csv` file for
node colors and sizes. Use `--shap-values-path`, `--plot-output-path`, and
`--plot-title` to override the defaults.

### 6. Summary feature heatmap

Draw a heatmap with standardized dataset summary characteristics:

```bash
poetry run python src/viz/plot_shap_summary_heatmaps.py
```

The script reads datasets from `data/db/summary.csv`. Outputs are saved to
`results/shap_summary_heatmaps/`.
