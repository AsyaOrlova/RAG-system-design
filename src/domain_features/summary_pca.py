from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
DATA_DB_DIR = DATA_DIR / "db"


FEATURE_GROUPS = {
    "Statistical": [
        "num_documents",
        "avg_doc_length",
        "std_doc_length",
        "total_corpus_size",
        "total_corpus_size_chars",
        "avg_doc_length_chars",
        "std_doc_length_chars",
    ],
    "Structural": [
        "table_density",
        "avg_tables_per_doc",
        "figure_density",
        "avg_figures_per_doc",
        "avg_sections_per_doc",
        "avg_references_count",
    ],
    "Linguistic": [
        "avg_sentence_length",
        "vocabulary_size",
        "technical_term_density",
        "formula_density",
        "chemical_formula_density",
        "chemical_name_density",
    ],
    "Embedding": [
        "embedding_variance",
        "avg_pairwise_similarity",
        "embedding_cluster_count",
        "embedding_silhouette_score",
    ],
}


def aggregate_by_pca(df, feature_groups, scale="zscore"):
    result = pd.DataFrame(index=df.index)
    loadings = {}

    for group_name, cols in feature_groups.items():
        x = df[cols].copy()

        if scale == "zscore":
            x_scaled = StandardScaler().fit_transform(x)
        elif scale == "minmax":
            x_scaled = MinMaxScaler().fit_transform(x)
        elif scale is None:
            x_scaled = x.values
        else:
            raise ValueError("scale must be 'zscore', 'minmax', or None")

        pca = PCA(n_components=1)
        result[group_name] = pca.fit_transform(x_scaled).ravel()

        loadings[group_name] = pd.Series(
            pca.components_[0],
            index=cols,
            name=f"{group_name}_PC1_loading",
        )

        if loadings[group_name].sum() < 0:
            result[group_name] *= -1
            loadings[group_name] *= -1

    return result, loadings


def dumbbell_plot_vertical(df, name, size_x=3, size_y=4):
    datasets = df.index.tolist()
    dataset_names = ["MC", "NZ", "OA"]
    features = df.columns.tolist()
    x = np.arange(len(features))

    plt.figure(figsize=(size_x, size_y))

    colors = ["tab:blue", "tab:green", "tab:red"]

    for i, feature in enumerate(features):
        values = df.loc[:, feature]

        plt.plot(
            [i, i],
            [values.min(), values.max()],
            color="lightgray",
            linewidth=2,
            zorder=1,
        )

        for j, dataset in enumerate(datasets):
            plt.scatter(
                i,
                values[dataset],
                color=colors[j],
                label=dataset_names[j] if i == 0 else "",
                s=100,
                zorder=2,
            )

    plt.xticks(x, features, rotation=30, ha="right")
    plt.ylabel("PCA aggregated value")
    plt.title("Dumbbell plot (grouped PCA features)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        SCRIPT_DIR / f"dumbbell_plot_vertical_{name}.svg",
        dpi=500,
        bbox_inches="tight",
    )
    plt.close()


def main():
    df = pd.read_csv(DATA_DB_DIR / "summary.csv")

    df_clean = df.copy()
    df_clean.drop(
        columns=[
            "dataset",
            "articles_file",
            "chunks_file",
            "documents_with_embeddings",
        ],
        inplace=True,
    )

    df_clean_norm = pd.DataFrame(
        MinMaxScaler().fit_transform(df_clean),
        index=df["dataset"],
        columns=df_clean.columns,
    )

    df_grouped_pca, _pca_loadings = aggregate_by_pca(
        df,
        FEATURE_GROUPS,
        scale="minmax",
    )
    df_grouped_pca.index = df["dataset"]

    df_grouped_pca_final = df_grouped_pca.copy()
    df_grouped_pca_final["dataset"] = df_grouped_pca_final.index
    df_grouped_pca_final.to_csv(DATA_DB_DIR / "PCA_features.csv", index=False)

    scaler = MinMaxScaler()
    df_pca_norm = pd.DataFrame(
        scaler.fit_transform(df_grouped_pca),
        index=df_grouped_pca.index,
        columns=df_grouped_pca.columns,
    )

    dumbbell_plot_vertical(df_pca_norm, "pca")
    dumbbell_plot_vertical(df_clean_norm, "all", size_x=12)


if __name__ == "__main__":
    main()
