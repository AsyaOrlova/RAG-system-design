import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAG_BEST_DIR = PROJECT_ROOT / "data" / "rag_best"
TOP_K_VALUES = [1, 3, 5, 10]
PARAMETER_COLUMNS = [
    "chunk_size",
    "chunk_overlap",
    "dense_k",
    "sparse_k",
    "search_mode",
    "reranker",
    "llm_model",
]
DATASET_FILE_ALIASES = {
    "complexes": "complexes",
    "nanozymes": "nano",
    "nano": "nano",
    "oxazo": "oxazo",
}


def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def normalize_value(value):
    """Normalize values before exact parameter matching."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def normalize_params(params):
    normalized = {
        column: normalize_value(params.get(column))
        for column in PARAMETER_COLUMNS
        if column in params
    }
    if "chunk_size" in normalized and "chunk_overlap" not in normalized:
        normalized["chunk_overlap"] = int(round(normalized["chunk_size"] * 0.25))

    return normalized


def params_match(left, right):
    left = normalize_params(left)
    right = normalize_params(right)
    return all(left.get(column) == right.get(column) for column in PARAMETER_COLUMNS)


def find_rank(best_params, top_rows):
    for index, row in enumerate(top_rows, start=1):
        if params_match(best_params, row):
            return index

    return None


def get_dataset_file_alias(dataset):
    if dataset not in DATASET_FILE_ALIASES:
        raise ValueError(
            f"Unsupported target dataset '{dataset}'. "
            f"Known datasets: {sorted(DATASET_FILE_ALIASES)}"
        )

    return DATASET_FILE_ALIASES[dataset]


def resolve_default_top_files(target_dataset, rag_best_dir=DEFAULT_RAG_BEST_DIR):
    alias = get_dataset_file_alias(target_dataset)
    rag_best_dir = Path(rag_best_dir)
    return [
        ("2_metrics", rag_best_dir / f"top10_{alias}.json"),
        ("6_metrics", rag_best_dir / f"top10_{alias}_6_metrics.json"),
    ]


def resolve_top_files(args, target_dataset):
    if args.top_file:
        resolved = []
        for top_file in args.top_file:
            label = Path(top_file).stem
            if label.startswith("top10_"):
                label = label[len("top10_") :]
            resolved.append((label, Path(top_file)))
        return resolved

    return resolve_default_top_files(target_dataset, args.rag_best_dir)


def build_result_row(label, top_file, best_params, top_rows):
    rank = find_rank(best_params, top_rows)
    row = {
        "top_list": label,
        "rank": rank if rank is not None else "",
        "in_top_1": rank is not None and rank <= 1,
        "in_top_3": rank is not None and rank <= 3,
        "in_top_5": rank is not None and rank <= 5,
        "in_top_10": rank is not None and rank <= 10,
    }
    row.update({f"best_{column}": best_params.get(column, "") for column in PARAMETER_COLUMNS})
    return row


def evaluate_experiment(experiment_dir, rag_best_dir=DEFAULT_RAG_BEST_DIR, top_files=None):
    experiment_dir = Path(experiment_dir)
    best_parameters_path = experiment_dir / "best_parameters.json"
    if not best_parameters_path.exists():
        raise FileNotFoundError(f"Missing best parameters file: {best_parameters_path}")

    best_parameters = load_json(best_parameters_path)
    target_dataset = best_parameters.get("target_dataset")
    if not target_dataset:
        raise ValueError(f"{best_parameters_path} does not contain 'target_dataset'")

    best_params = normalize_params(best_parameters["params"])
    top_file_entries = (
        [(Path(path).stem, Path(path)) for path in top_files]
        if top_files
        else resolve_default_top_files(target_dataset, rag_best_dir)
    )

    rows = []
    for label, top_file in top_file_entries:
        if not top_file.exists():
            raise FileNotFoundError(f"Missing top-list file: {top_file}")
        top_rows = load_json(top_file)
        rows.append(build_result_row(label, top_file, best_params, top_rows))

    return rows


def save_rows(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "top_list",
        "rank",
        "in_top_1",
        "in_top_3",
        "in_top_5",
        "in_top_10",
        *[f"best_{column}" for column in PARAMETER_COLUMNS],
    ]
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check whether optimized best parameters are present in top-1, top-3, "
            "top-5, and top-10 RAG configurations for the target dataset."
        )
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Optimization experiment directory containing best_parameters.json.",
    )
    parser.add_argument(
        "--rag-best-dir",
        type=Path,
        default=DEFAULT_RAG_BEST_DIR,
        help=f"Directory with top10 JSON files. Defaults to {DEFAULT_RAG_BEST_DIR}",
    )
    parser.add_argument(
        "--top-file",
        action="append",
        type=Path,
        help=(
            "Explicit top-list JSON file. Can be passed multiple times. "
            "If omitted, files are selected by target_dataset."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-name",
        default="top_hit_metrics.csv",
        help="CSV filename saved inside the experiment directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = evaluate_experiment(
        experiment_dir=args.experiment_dir,
        rag_best_dir=args.rag_best_dir,
        top_files=args.top_file,
    )
    output_path = save_rows(rows, Path(args.experiment_dir) / args.output_name)

    print(f"Saved top-hit metrics to {output_path}")
    for row in rows:
        print(
            f"{row['top_list']}: "
            f"rank={row['rank'] or 'not in top-10'}, "
            f"in_top_1={row['in_top_1']}, "
            f"in_top_3={row['in_top_3']}, "
            f"in_top_5={row['in_top_5']}, "
            f"in_top_10={row['in_top_10']}"
        )


if __name__ == "__main__":
    main()
