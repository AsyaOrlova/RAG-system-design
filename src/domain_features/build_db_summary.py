import ast
import csv
import os
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DB_DIR = PROJECT_ROOT / "data" / "db"
SUMMARY_PATH = DATA_DB_DIR / "summary.csv"
RANDOM_SEED = 42

IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)|<img\b", re.IGNORECASE)
IMAGE_PATH_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)", re.IGNORECASE)
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", re.MULTILINE)
TABLE_HTML_RE = re.compile(r"<table\b", re.IGNORECASE)
TABLE_MD_RE = re.compile(
    r"^\s*\|?.+\|.+\n\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$",
    re.MULTILINE,
)
REFERENCE_ENTRY_RE = re.compile(
    r"^\s*(?:\[\d+\]|\d+\.\s+|\d+\)\s+|-\s+|\*\s+)",
    re.MULTILINE,
)
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", re.IGNORECASE)
LATEX_RE = re.compile(
    r"\$[^$\n]+\$"
    r"|\\\([^)]+\\\)"
    r"|\\\[[^\]]+\\\]"
    r"|\\[A-Za-z]+(?:\{[^{}]*\})?"
)
FIGURE_CAPTION_RE = re.compile(
    r"(?im)^\s*(?:\*\*)?(?:figure|fig\.?)\s*\d+[a-z]?(?:\s*[\.\):\-]|\s+)"
)
INLINE_TAG_RE = re.compile(r"<[^>]+>")
PAGE_ARTIFACT_IMAGE_RE = re.compile(r"_page_\d+_picture_\d+\.(?:png|jpg|jpeg|webp)\b", re.IGNORECASE)

METADATA_HEADINGS = {
    "authors",
    "author",
    "affiliations",
    "affiliation",
    "keywords",
    "key words",
    "highlights",
    "graphical abstract",
    "supplementary material",
    "supplementary materials",
    "supplementary information",
    "funding",
    "declarations",
    "conflicts of interest",
    "conflict of interest",
    "author contributions",
    "author contribution",
    "acknowledgments",
    "acknowledgements",
    "address correspondence to",
}
REFERENCE_HEADINGS = {"references", "reference", "bibliography"}
SCISPACY_MODEL = "en_core_sci_sm"
SCISPACY_TERM_NER_MODEL = "en_ner_bionlp13cg_md"
SCISPACY_CHEMICAL_NER_MODEL = "en_ner_bc5cdr_md"
FORMULA_TOKEN_RE = re.compile(
    r"^(?:"
    r"(?:[A-Z][a-z]?\d*){2,}"
    r"|[A-Z][a-z]?-\d{1,3}[A-Za-z]?"
    r"|(?:[A-Z][a-z]?\d*)+\([A-Za-z0-9]+\)\d*"
    r")$"
)


def safe_mean(values):
    return float(np.mean(values)) if values else float("nan")


def safe_std(values):
    return float(np.std(values, ddof=0)) if values else float("nan")


def log(message):
    tqdm.write(message)


@lru_cache(maxsize=1)
def get_nlp():
    try:
        return spacy.load(SCISPACY_MODEL)
    except OSError as exc:
        raise RuntimeError(
            "scispaCy model is required. Install with "
            "`pip install scispacy spacy` and "
            "`pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz`."
        ) from exc


@lru_cache(maxsize=1)
def get_term_ner_nlp():
    try:
        return spacy.load(SCISPACY_TERM_NER_MODEL)
    except OSError as exc:
        raise RuntimeError(
            "scispaCy term NER model is required. Install with "
            "`pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bionlp13cg_md-0.5.4.tar.gz`."
        ) from exc


@lru_cache(maxsize=1)
def get_chemical_ner_nlp():
    try:
        return spacy.load(SCISPACY_CHEMICAL_NER_MODEL)
    except OSError as exc:
        raise RuntimeError(
            "scispaCy chemical NER model is required. Install with "
            "`pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz`."
        ) from exc


@lru_cache(maxsize=1)
def ensure_nltk_sentence_tokenizer():
    try:
        sent_tokenize("Test sentence. Another one.", language="english")
    except LookupError as exc:
        raise RuntimeError(
            "NLTK punkt data is required. Install it with "
            "`python -m nltk.downloader punkt punkt_tab`."
        ) from exc
    return True


def iter_text_tokens(doc):
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        text = token.text.strip()
        if text:
            yield text


def iter_lemmas(doc):
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        lemma = token.lemma_.strip().lower()
        if not lemma or lemma == "-pron-":
            lemma = token.text.strip().lower()
        if lemma:
            yield lemma


def strip_markdown(text):
    cleaned = INLINE_TAG_RE.sub(" ", text or "")
    cleaned = re.sub(r"[*_`~]+", " ", cleaned)
    cleaned = re.sub(r"\[(.*?)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def normalize_heading(heading):
    cleaned = strip_markdown(heading).strip(" :.-").lower()
    cleaned = re.sub(r"^\d+(?:\.\d+)*\s+", "", cleaned)
    return cleaned


def iter_headings(text):
    for match in HEADING_RE.finditer(text or ""):
        yield match.start(), match.end(), match.group(1), normalize_heading(match.group(1))


def normalize_for_nlp(text):
    normalized = re.sub(r"(?<!\n)\n(?!\n)", " ", text or "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def count_tables(text):
    html_count = len(TABLE_HTML_RE.findall(text))
    md_count = len(TABLE_MD_RE.findall(text))
    return html_count + md_count


def count_sections(text):
    count = 0
    for _, _, raw_heading, normalized_heading in iter_headings(text):
        if not normalized_heading:
            continue
        if normalized_heading in METADATA_HEADINGS:
            continue
        if re.fullmatch(r"\d+(?:\.\d+)*", normalized_heading):
            continue
        if len(normalized_heading) <= 2:
            continue
        if re.fullmatch(r"[a-z]\b", normalized_heading):
            continue
        if normalized_heading.startswith("address correspondence to"):
            continue
        count += 1
    return count


def count_figures(text):
    caption_count = len(FIGURE_CAPTION_RE.findall(text or ""))
    image_paths = IMAGE_PATH_RE.findall(text or "")
    non_artifact_images = [
        path for path in image_paths if not PAGE_ARTIFACT_IMAGE_RE.search(path)
    ]
    html_images = [
        match.group(0)
        for match in IMAGE_RE.finditer(text or "")
        if match.group(0).lower().startswith("<img")
    ]
    return max(caption_count, len(non_artifact_images) + len(html_images))


def extract_references_section(text):
    headings = list(iter_headings(text))
    for index, (_, end, _, normalized_heading) in enumerate(headings):
        if normalized_heading not in REFERENCE_HEADINGS:
            continue
        start = end
        next_start = headings[index + 1][0] if index + 1 < len(headings) else len(text or "")
        return (text or "")[start:next_start]
    return ""


def count_reference_entries(text):
    references = extract_references_section(text)
    if not references.strip():
        return 0

    entry_count = len(REFERENCE_ENTRY_RE.findall(references))
    if entry_count:
        return entry_count

    doi_count = len(DOI_RE.findall(references))
    if doi_count:
        return doi_count

    blocks = [block.strip() for block in re.split(r"\n\s*\n", references) if block.strip()]
    return len(blocks)


def parse_embedding(raw_value):
    if not raw_value:
        return None

    try:
        vector = np.asarray(ast.literal_eval(raw_value), dtype=float)
    except (ValueError, SyntaxError):
        return None

    if vector.ndim != 1 or vector.size == 0:
        return None
    return vector


def is_formula_like(text):
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if LATEX_RE.fullmatch(cleaned):
        return True
    return bool(FORMULA_TOKEN_RE.fullmatch(cleaned))


def align_ent_to_doc(base_doc, ent):
    return base_doc.char_span(ent.start_char, ent.end_char, alignment_mode="expand")


def compute_density_sets(base_doc, term_ner_doc, chemical_ner_doc):
    word_token_ids = {token.i for token in base_doc if token.is_alpha}
    lexical_token_ids = {
        token.i for token in base_doc if not token.is_space and not token.is_punct
    }

    filtered_term_spans = []
    for ent in term_ner_doc.ents:
        span = align_ent_to_doc(base_doc, ent)
        if span is None:
            continue

        span_text = span.text.strip()
        if is_formula_like(span_text):
            continue

        filtered_term_spans.append(span)

    term_token_ids = {
        token.i
        for span in filtered_term_spans
        for token in span
        if token.is_alpha
    }

    chemical_span_token_ids = set()
    for ent in chemical_ner_doc.ents:
        if ent.label_ != "CHEMICAL":
            continue
        span = align_ent_to_doc(base_doc, ent)
        if span is None:
            continue
        chemical_span_token_ids.update(token.i for token in span if token.is_alpha)

    formula_token_ids = {
        token.i for token in base_doc if is_formula_like(token.text)
    }

    technical_term_token_ids = term_token_ids - formula_token_ids
    chemical_name_token_ids = chemical_span_token_ids - formula_token_ids
    chemical_formula_token_ids = formula_token_ids & lexical_token_ids

    denominator = max(len(word_token_ids), 1)
    lexical_denominator = max(len(lexical_token_ids), 1)

    return {
        "technical_term_density": len(technical_term_token_ids) / denominator,
        "chemical_name_density": len(chemical_name_token_ids) / denominator,
        "chemical_formula_density": len(chemical_formula_token_ids) / lexical_denominator,
    }


def compute_article_metrics(article_rows):
    nlp = get_nlp()
    term_ner_nlp = get_term_ner_nlp()
    chemical_ner_nlp = get_chemical_ner_nlp()
    ensure_nltk_sentence_tokenizer()
    normalized_texts = [normalize_for_nlp((row.get("md", "") or "")) for row in article_rows]

    doc_lengths_chars = []
    doc_lengths_tokens = []
    section_counts = []
    reference_counts = []
    table_docs = 0
    table_counts = []
    figure_docs = 0
    figure_counts = []
    total_tokens = 0
    sentence_length_means = []
    vocabulary_lemmas = set()
    technical_term_densities = []
    chemical_name_densities = []
    chemical_formula_densities = []
    formula_densities = []
    total_chars = 0

    article_iter = zip(
        article_rows,
        normalized_texts,
        nlp.pipe(normalized_texts, batch_size=8),
        term_ner_nlp.pipe(normalized_texts, batch_size=8),
        chemical_ner_nlp.pipe(normalized_texts, batch_size=8),
    )

    for row, normalized_text, doc, term_ner_doc, chemical_ner_doc in tqdm(
        article_iter,
        total=len(article_rows),
        desc="  articles",
        leave=False,
    ):
        text = row.get("md", "") or ""
        raw_tokens = list(iter_text_tokens(doc))
        lemmas = list(iter_lemmas(doc))
        sentence_lengths = []
        for sentence in sent_tokenize(normalized_text, language="english"):
            sent_doc = nlp.make_doc(sentence)
            sent_tokens = list(iter_text_tokens(sent_doc))
            if sent_tokens:
                sentence_lengths.append(len(sent_tokens))
        tables_in_doc = count_tables(text)
        refs_in_doc = count_reference_entries(text)
        sections_in_doc = count_sections(text)
        figures_in_doc = count_figures(text)
        formula_matches = LATEX_RE.findall(text)
        formula_chars = sum(len(match) for match in formula_matches)
        token_count = len(raw_tokens)
        densities = compute_density_sets(doc, term_ner_doc, chemical_ner_doc)

        doc_lengths_chars.append(len(text))
        doc_lengths_tokens.append(token_count)
        section_counts.append(sections_in_doc)
        reference_counts.append(refs_in_doc)
        table_counts.append(tables_in_doc)
        figure_counts.append(figures_in_doc)
        total_chars += len(text)
        total_tokens += token_count
        vocabulary_lemmas.update(lemmas)
        if sentence_lengths:
            sentence_length_means.append(float(np.mean(sentence_lengths)))
        technical_term_densities.append(densities["technical_term_density"])
        chemical_name_densities.append(densities["chemical_name_density"])
        chemical_formula_densities.append(densities["chemical_formula_density"])
        if text:
            formula_densities.append(formula_chars / len(text))

        if tables_in_doc > 0:
            table_docs += 1
        if figures_in_doc > 0:
            figure_docs += 1

    doc_count = len(article_rows)
    vocabulary_size = len(vocabulary_lemmas)

    return {
        "num_documents": doc_count,
        "avg_doc_length": safe_mean(doc_lengths_tokens),
        "std_doc_length": safe_std(doc_lengths_tokens),
        "total_corpus_size": total_tokens,
        "avg_doc_length_chars": safe_mean(doc_lengths_chars),
        "std_doc_length_chars": safe_std(doc_lengths_chars),
        "total_corpus_size_chars": total_chars,
        "table_density": (table_docs / doc_count) if doc_count else float("nan"),
        "avg_tables_per_doc": safe_mean(table_counts),
        "figure_density": (figure_docs / doc_count) if doc_count else float("nan"),
        "avg_figures_per_doc": safe_mean(figure_counts),
        "avg_sections_per_doc": safe_mean(section_counts),
        "avg_references_count": safe_mean(reference_counts),
        "avg_sentence_length": safe_mean(sentence_length_means),
        "vocabulary_size": vocabulary_size,
        "technical_term_density": safe_mean(technical_term_densities),
        "chemical_name_density": safe_mean(chemical_name_densities),
        "chemical_formula_density": safe_mean(chemical_formula_densities),
        "formula_density": safe_mean(formula_densities),
    }


def build_document_embeddings(chunk_rows):
    vectors_by_article = defaultdict(list)

    for row in tqdm(chunk_rows, total=len(chunk_rows), desc="  chunks", leave=False):
        vector = parse_embedding(row.get("embedding", ""))
        if vector is None:
            continue
        vectors_by_article[row.get("article_key", "")].append(vector)

    document_vectors = []
    for vectors in vectors_by_article.values():
        if not vectors:
            continue
        mean_vector = np.mean(np.vstack(vectors), axis=0)
        document_vectors.append(mean_vector)

    if not document_vectors:
        return np.empty((0, 0), dtype=float)

    return normalize_rows(np.vstack(document_vectors))


def sample_embeddings(embeddings, max_samples, rng):
    if len(embeddings) <= max_samples:
        return embeddings
    indices = rng.choice(len(embeddings), size=max_samples, replace=False)
    return embeddings[indices]


def normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float64, order="C")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    return np.asarray(normalized, dtype=np.float64, order="C")


def compute_avg_pairwise_similarity(embeddings, rng, max_pairs=20000, max_docs=4000):
    if len(embeddings) < 2:
        return float("nan")

    similarity_matrix = cosine_similarity(embeddings)
    upper = similarity_matrix[np.triu_indices(len(embeddings), k=1)]
    if upper.size == 0:
        return float("nan")
    similarities = upper
    return float(np.mean(similarities))


def compute_embedding_cluster_count(embeddings, rng, max_docs=2000, max_k=8):
    if len(embeddings) == 0:
        return float("nan")
    if len(embeddings) < 3:
        return 1.0, None, None

    sampled = sample_embeddings(embeddings, max_docs, rng)
    sampled = np.asarray(sampled, dtype=np.float64, order="C")
    upper_k = min(max_k, len(sampled) - 1)
    if upper_k < 2:
        return 1.0, None, None

    ks = []
    inertias = []
    for k in tqdm(range(1, upper_k + 1), desc="  kmeans", leave=False):
        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=RANDOM_SEED,
            algorithm="lloyd",
        )
        model.fit(sampled)
        ks.append(k)
        inertias.append(float(model.inertia_ / len(sampled)))

    if len(ks) < 3:
        final_k = float(ks[-1])
        final_model = KMeans(
            n_clusters=int(final_k),
            n_init=10,
            random_state=RANDOM_SEED,
            algorithm="lloyd",
        )
        labels = final_model.fit_predict(sampled)
        return final_k, labels, sampled

    points = np.column_stack([ks, inertias])
    start = points[0]
    end = points[-1]
    line_vector = end - start
    line_norm = np.linalg.norm(line_vector)
    if line_norm == 0:
        return float(ks[0])

    distances = []
    for point in points:
        delta = point - start
        numerator = abs(line_vector[0] * delta[1] - line_vector[1] * delta[0])
        distances.append(float(numerator / line_norm))

    elbow_index = int(np.argmax(distances))
    final_k = float(ks[elbow_index])
    final_model = KMeans(
        n_clusters=int(final_k),
        n_init=10,
        random_state=RANDOM_SEED,
        algorithm="lloyd",
    )
    labels = final_model.fit_predict(sampled)
    return final_k, labels, sampled


def compute_embedding_metrics(chunk_rows):
    rng = np.random.default_rng(RANDOM_SEED)
    document_embeddings = build_document_embeddings(chunk_rows)

    if len(document_embeddings) == 0:
        return {
            "documents_with_embeddings": 0,
            "embedding_variance": float("nan"),
            "avg_pairwise_similarity": float("nan"),
            "embedding_cluster_count": float("nan"),
            "embedding_silhouette_score": float("nan"),
        }

    cluster_count, labels, clustered_embeddings = compute_embedding_cluster_count(
        document_embeddings, rng
    )
    if (
        labels is not None
        and clustered_embeddings is not None
        and len(set(labels)) > 1
        and len(clustered_embeddings) == len(labels)
    ):
        sil_score = float(silhouette_score(clustered_embeddings, labels))
    else:
        sil_score = float("nan")

    return {
        "documents_with_embeddings": int(len(document_embeddings)),
        "embedding_variance": float(np.var(document_embeddings, axis=0).mean()),
        "avg_pairwise_similarity": compute_avg_pairwise_similarity(document_embeddings, rng),
        "embedding_cluster_count": cluster_count,
        "embedding_silhouette_score": sil_score,
    }


def read_csv_rows(path):
    frame = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False)
    return frame.to_dict(orient="records")


def discover_dataset_files():
    datasets = []
    for dataset_dir in sorted(DATA_DB_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue

        article_files = sorted(dataset_dir.glob("articles_*.csv"))
        chunk_files = sorted(dataset_dir.glob("chunks_*.csv"))
        if not article_files or not chunk_files:
            continue

        datasets.append(
            {
                "dataset": dataset_dir.name,
                "articles_path": article_files[-1],
                "chunks_path": chunk_files[-1],
            }
        )

    return datasets


def build_summary_rows():
    rows = []
    datasets = discover_dataset_files()
    log(f"Found {len(datasets)} dataset(s) in {DATA_DB_DIR}")

    for dataset in tqdm(datasets, desc="datasets"):
        log("")
        log(f"[{dataset['dataset']}] Loading files")
        log(f"  articles: {dataset['articles_path'].name}")
        log(f"  chunks:   {dataset['chunks_path'].name}")
        article_rows = read_csv_rows(dataset["articles_path"])
        chunk_rows = read_csv_rows(dataset["chunks_path"])
        log(
            f"  loaded {len(article_rows)} article rows and {len(chunk_rows)} chunk rows"
        )

        log(f"[{dataset['dataset']}] Computing article metrics")
        metrics = {}
        metrics.update(compute_article_metrics(article_rows))
        log(
            "  article metrics:"
            f" docs={metrics['num_documents']},"
            f" avg_tokens={metrics['avg_doc_length']:.1f},"
            f" avg_sent_len={metrics['avg_sentence_length']:.2f},"
            f" vocab={metrics['vocabulary_size']},"
            f" avg_sections={metrics['avg_sections_per_doc']:.2f},"
            f" avg_refs={metrics['avg_references_count']:.2f},"
            f" avg_figures={metrics['avg_figures_per_doc']:.2f},"
            f" tech_density={metrics['technical_term_density']:.4f},"
            f" chem_name_density={metrics['chemical_name_density']:.4f},"
            f" chem_formula_density={metrics['chemical_formula_density']:.4f},"
            f" latex_density={metrics['formula_density']:.4f}"
        )

        log(f"[{dataset['dataset']}] Computing embedding metrics")
        metrics.update(compute_embedding_metrics(chunk_rows))
        log(
            "  embedding metrics:"
            f" docs_with_embeddings={metrics['documents_with_embeddings']},"
            f" variance={metrics['embedding_variance']:.6f},"
            f" pairwise_sim={metrics['avg_pairwise_similarity']:.4f},"
            f" clusters={metrics['embedding_cluster_count']},"
            f" silhouette={metrics['embedding_silhouette_score']:.4f}"
        )
        metrics.update(
            {
                "dataset": dataset["dataset"],
                "articles_file": dataset["articles_path"].name,
                "chunks_file": dataset["chunks_path"].name,
            }
        )
        rows.append(metrics)

    return rows


def write_summary(rows, output_path=SUMMARY_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No datasets found in data/db")

    fieldnames = [
        "dataset",
        "articles_file",
        "chunks_file",
        "num_documents",
        "documents_with_embeddings",
        "avg_doc_length",
        "std_doc_length",
        "total_corpus_size",
        "avg_doc_length_chars",
        "std_doc_length_chars",
        "total_corpus_size_chars",
        "table_density",
        "avg_tables_per_doc",
        "figure_density",
        "avg_figures_per_doc",
        "avg_sections_per_doc",
        "avg_references_count",
        "avg_sentence_length",
        "vocabulary_size",
        "technical_term_density",
        "chemical_name_density",
        "chemical_formula_density",
        "formula_density",
        "embedding_variance",
        "avg_pairwise_similarity",
        "embedding_cluster_count",
        "embedding_silhouette_score",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    log(f"Using scispaCy model: {SCISPACY_MODEL}")
    log(f"Using scispaCy term NER model: {SCISPACY_TERM_NER_MODEL}")
    log(f"Using scispaCy chemical NER model: {SCISPACY_CHEMICAL_NER_MODEL}")
    log("Using NLTK sentence tokenizer: punkt")
    rows = build_summary_rows()
    write_summary(rows)
    log(f"Summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
