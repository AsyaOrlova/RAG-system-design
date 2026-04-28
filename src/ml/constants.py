from pathlib import Path

##### paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREPROCESS_RESULTS_DIR = PROJECT_ROOT / "data" / "processed"
ML_RESULTS_DIR = PROJECT_ROOT / "results"
SUMMARY_PATH = PROJECT_ROOT / "data" / "db" / "summary.csv"
MODEL_SELECTION_METRICS_PATH = ML_RESULTS_DIR / "train_models" / "model_selection_metrics.csv"

##### columns

TARGET_COLUMNS = [
    "bert_score_recall",
    "cosine_similarity",
    "golden_doi_mrr",
    "golden_doi_recall",
    "numeric_recall",
    "rouge_l_recall",
    "scientific_fact_recall",
    "source_support_hit",
    "source_support_max_similarity",
    "token_f1",
    "token_recall",
]

DROP_COLUMNS = [
    "corpus",
    "kb_alias",
    "eval_alias",
    "count_tokens",
    "paragraph_filtrator",
    "table_grabber",
]

SUMMARY_METADATA_COLUMNS = {"dataset", "articles_file", "chunks_file", "num_documents"}

##### random states

TRAIN_MODELS_RANDOM_STATE = 42
