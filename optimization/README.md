# Optimization Pipeline

This folder contains the surrogate-model optimization workflow for discrete RAG
parameters. The optimizer uses a saved surrogate `.pkl` model to predict target
metrics for sampled parameter configurations, then writes the best predicted
configuration and experiment metadata.

## Inputs

- A trained surrogate model, usually from `results/<dataset>/hp_tuning/best_tuned_model.pkl`.
- A target dataset name passed with `--target-dataset`.
- Dataset summary features from `data/db/summary.csv` by default.
- A parameter-space JSON from `optimization/parameter_spaces/`.

## Run Optimization

Optimize the default objective metrics (`golden_doi_recall` and
`scientific_fact_recall`) for one target dataset:

```bash
poetry run python optimization/optimize.py \
  -m results/oxazo_no_summary/hp_tuning/best_tuned_model.pkl \
  --target-dataset oxazo \
  -n 100
```

Choose metrics explicitly:

```bash
poetry run python optimization/optimize.py \
  -m results/oxazo_no_summary/hp_tuning/best_tuned_model.pkl \
  --target-dataset oxazo \
  --metrics golden_doi_recall scientific_fact_recall rouge_l_recall \
  --metric-weights 0.5 0.3 0.2 \
  -n 200
```

Run multi-objective optimization and store Pareto-front representatives:

```bash
poetry run python optimization/optimize.py \
  -m results/oxazo_no_summary/hp_tuning/best_tuned_model.pkl \
  --target-dataset oxazo \
  --metrics golden_doi_recall scientific_fact_recall \
  --multi-objective \
  -n 200
```

Outputs are saved under `optimization/experiments/<experiment_name>/`, including:

- `best_parameters.json`
- `trials.csv`
- `study_summary.json`

## Metric Filters

You can restrict final best-configuration selection to trials that meet preset
thresholds for support metrics:

```bash
poetry run python optimization/optimize.py \
  -m results/oxazo_no_summary/hp_tuning/best_tuned_model.pkl \
  --target-dataset oxazo \
  --filter-best-metrics \
  -n 200
```

Available filter modes:

- `--filter-best-metrics`: dataset-specific best thresholds.
- `--filter-top10-best-metrics`: dataset-specific top-10 thresholds.
- `--filter-default-best-metrics`: default thresholds.

Use only one filter mode per run.

## Evaluate Top-Hit Matches

Check whether an optimized configuration appears in empirical top lists from
`data/rag_best/`:

```bash
poetry run python optimization/evaluate_top_hits.py \
  optimization/experiments/<experiment_name>
```

This writes `top_hit_metrics.csv` inside the experiment directory.

## Summarize Experiments

Collect all optimization experiments into one table:

```bash
poetry run python optimization/summarize_experiments.py
```

By default, the summary is saved to:

```text
optimization/experiments/experiments_summary.csv
```

## Parameter Spaces

Parameter-space files live in `optimization/parameter_spaces/`. Each parameter
can be either a list of allowed values or an object with:

- `values`: allowed discrete values.
- `default`: value used as the default configuration.
- `sampling_weight`: probability of sampling from the allowed values instead of
  using the default during optimization.

The optimizer currently covers:

- `chunk_size`
- `dense_k`
- `sparse_k`
- `search_mode`
- `reranker`
- `llm_model`

`chunk_overlap` is derived as `25%` of `chunk_size`.

