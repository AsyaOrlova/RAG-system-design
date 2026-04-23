# `summary.csv`

`data/db/summary.csv` собирается скриптом `src/build_db_summary.py`. Для каждой подпапки `data/db/<dataset>/` берется актуальная пара `articles_*.csv` и `chunks_*.csv`, после чего в итоговую таблицу записывается одна строка на датасет.

## Источники

- `articles_*.csv`: `article_key`, `md`; `md` считается основным текстом статьи.
- `chunks_*.csv`: `article_key`, `embedding`; chunk embeddings усредняются по статье, затем document embedding L2-нормализуется.

## Колонки

- `dataset`, `articles_file`, `chunks_file`: идентификатор датасета и имена файлов, по которым считались метрики.
- `num_documents`: число статей в `articles_*.csv`.
- `documents_with_embeddings`: число статей, для которых удалось собрать хотя бы один валидный embedding.

### Размер корпуса

- `avg_doc_length`, `std_doc_length`, `total_corpus_size`: средняя длина статьи, стандартное отклонение и суммарный размер корпуса в токенах; токены берутся через `scispaCy` (`en_core_sci_sm`) без пробелов и пунктуации.
- `avg_doc_length_chars`, `std_doc_length_chars`, `total_corpus_size_chars`: те же метрики в символах, считались по `len(md)`.

### Структура

- `table_density`: доля документов, где найдена хотя бы одна таблица; ищутся markdown-таблицы и HTML `<table>`.
- `avg_tables_per_doc`: среднее число найденных таблиц на документ.
- `figure_density`: доля документов с рисунками; используются подписи `Figure/Fig.` и неартефактные markdown/HTML-изображения.
- `avg_figures_per_doc`: среднее число рисунков на документ по тому же счетчику.
- `avg_sections_per_doc`: среднее число markdown-заголовков после фильтрации служебных и артефактных heading'ов.
- `avg_references_count`: среднее число ссылок в секции `References/Reference/Bibliography`; если явной нумерации нет, используются DOI или абзацы как fallback.

### Лингвистика

- `avg_sentence_length`: для каждой статьи предложения выделяются через `nltk.sent_tokenize`, затем считается средняя длина предложения в scispaCy-токенах; по датасету усредняется уже это значение на уровне статьи.
- `vocabulary_size`: размер множества уникальных лемм корпуса после токенизации и лемматизации в `scispaCy`.
- `technical_term_density`: для каждой статьи берутся entity spans из `en_ner_bionlp13cg_md`, из них исключаются formula-like spans, после чего считается доля алфавитных токенов терминов среди обычных словесных токенов статьи.
- `chemical_name_density`: доля токенов, попавших в `CHEMICAL` entities из `en_ner_bc5cdr_md`, без formula-like токенов.
- `chemical_formula_density`: доля formula-like токенов среди lexical tokens статьи; считается regex-эвристикой по токенам.
- `formula_density`: доля символов статьи, занятых LaTeX-фрагментами (`$...$`, `\(...\)`, `\[...\]`, LaTeX-команды); по датасету усредняется по статьям.

### Эмбеддинги

- `embedding_variance`: средняя дисперсия координат L2-нормализованных document embeddings.
- `avg_pairwise_similarity`: средняя косинусная близость по всем уникальным парам документов.
- `embedding_cluster_count`: число кластеров по elbow heuristic на `KMeans`.
- `embedding_silhouette_score`: silhouette score для того же разбиения `KMeans`, которое выбрано в `embedding_cluster_count`; чем ближе к `1`, тем лучше разделены кластеры, около `0` означает слабое разделение, отрицательные значения указывают на плохое разбиение.
