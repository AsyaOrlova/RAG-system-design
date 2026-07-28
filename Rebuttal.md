# Additional experiments for NIPS 2026 rebuttal

## 1. Experiments with leave-one-parameter-out dataset splits

Following the Reviewer's suggestion, we performed additional leave-one-parameter-value-out (LOPO) experiments for the numerical parameters (chunk_size, dense_k, and sparse_k) by iteratively holding out all configurations containing one parameter value and pooling predictions across folds. For dense_k and sparse_k, three folds were constructed by holding out one retrieval depth (5, 10, or 20) in each fold. Because dense_k=0 and sparse_k=0 denote inactive retrieval components rather than valid retrieval-depth values, these configurations were not treated as held-out parameter values. We did not apply LOPO to search mode, reranker, or LLM model, as these categorical variables would require predicting entirely unseen categories, which is not meaningful for tree-based surrogate models.

Predictions were generated only for the held-out configurations. After completing all folds, the out-of-fold predictions were concatenated so that every eligible configuration was predicted exactly once by a model that had not observed its corresponding parameter value during training. Model performance was then evaluated using a pooled coefficient of determination (R^2), computed over all concatenated out-of-fold predictions. 

### OA

| Split strategy | BERT Score Recall | Cosine Similarity | Golden DOI MRR | Golden DOI Recall | ROUGE-L Recall | Scientific Fact Recall |
|----------------|------------------:|------------------:|---------------:|------------------:|---------------:|-----------------------:|
| Random 80/20 | 0.84 | 0.84 | 0.91 | 0.96 | 0.85 | 0.95 |
| LOPO sparse_k | 0.78 | 0.73 | 0.77 | 0.66 | 0.68 | 0.91 |
| LOPO chunk_size | 0.76 | 0.73 | 0.87 | 0.93 | 0.74 | 0.94 |
| LOPO dense_k | 0.37 | 0.32 | 0.70 | 0.34 | 0.26 | 0.84 |

### MC

| Split strategy | BERT Score Recall | Cosine Similarity | Golden DOI MRR | Golden DOI Recall | ROUGE-L Recall | Scientific Fact Recall |
|----------------|------------------:|------------------:|---------------:|------------------:|---------------:|-----------------------:|
| Random 80/20 | 0.89 | 0.87 | 0.92 | 0.97 | 0.86 | 0.94 |
| LOPO sparse_k | 0.87 | 0.82 | 0.87 | 0.86 | 0.85 | 0.94 |
| LOPO chunk_size | 0.80 | 0.74 | 0.82 | 0.94 | 0.74 | 0.92 |
| LOPO dense_k | 0.54 | 0.34 | -2.30 | -2.30 | 0.65 | 0.87 |

### NZ

| Split strategy | BERT Score Recall | Cosine Similarity | Golden DOI MRR | Golden DOI Recall | ROUGE-L Recall | Scientific Fact Recall |
|----------------|------------------:|------------------:|---------------:|------------------:|---------------:|-----------------------:|
| Random 80/20 | 0.97 | 0.95 | 0.97 | 0.99 | 0.96 | 0.97 |
| LOPO sparse_k | 0.95 | 0.92 | 0.88 | 0.85 | 0.90 | 0.95 |
| LOPO chunk_size | 0.93 | 0.88 | 0.95 | 0.97 | 0.90 | 0.94 |
| LOPO dense_k | 0.74 | 0.26 | -7.02 | -3.52 | 0.23 | 0.71 |


Overall, the surrogate generalizes well to unseen values of chunk_size and sparse_k, with performance comparable to the random split across most metrics and domains. For dense_k, results are more mixed: while several metrics, particularly Scientific Fact Recall, remain consistently well predicted, some retrieval metrics show lower accuracy. This is consistent with our SHAP-IQ analysis, which identified dense_k as one of the most influential parameters, making LOPO a substantially more challenging extrapolation setting. Nevertheless, the surrogate retains strong performance across a substantial subset of metrics even under this stricter protocol, supporting our conclusion that it captures meaningful structure in the RAG design space rather than merely interpolating between neighbouring configurations.


## 2. Variability of metrics across questions and scenarios

Using one RAG configuration as an example, we analyzed the variability of RAG metrics across all question from OA benchmark, as well as the variability across scenarios.

| Metric | Overall | Aggregation | Comparison | Conditional Query | Direct Fact Retrieval | Multi-hop |
|--------|--------:|------------:|-----------:|------------------:|----------------------:|----------:|
| BERT Score Recall | 0.37 ± 0.17 | 0.40 ± 0.16 | 0.49 ± 0.19 | 0.28 ± 0.14 | 0.37 ± 0.17 | 0.32 ± 0.14 |
| Cosine Similarity | 0.51 ± 0.23 | 0.61 ± 0.22 | 0.71 ± 0.27 | 0.46 ± 0.13 | 0.44 ± 0.19 | 0.36 ± 0.18 |
| Golden DOI MRR | 0.58 ± 0.32 | 0.35 ± 0.27 | 0.69 ± 0.27 | 0.44 ± 0.24 | 1.00 ± 0.00 | 0.49 ± 0.28 |
| Golden DOI Recall | 0.78 ± 0.34 | 0.67 ± 0.41 | 0.83 ± 0.25 | 0.75 ± 0.35 | 1.00 ± 0.00 | 0.70 ± 0.42 |
| ROUGE-L Recall | 0.55 ± 0.37 | 0.45 ± 0.22 | 0.51 ± 0.23 | 0.72 ± 0.45 | 0.64 ± 0.40 | 0.45 ± 0.45 |
| Scientific Fact Recall | 0.93 ± 0.18 | 0.80 ± 0.26 | 0.88 ± 0.27 | 1.00 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 |

As expected, different question types exhibit different intrinsic difficulty. Direct fact retrieval achieves nearly perfect retrieval performance (Golden DOI Recall = 1.00, Scientific Fact Recall = 1.00), whereas multi-hop questions remain the most challenging. These results indicate that the benchmark contains a diverse spectrum of question complexities rather than being dominated by a single scenario, providing further confidence that the optimization results are representative across different QA tasks.


## 3. Positioning of our work against existing RAG hyperparameter optimization and AutoML-style surrogate optimization methods

We provide an additional table that compares out approach to the exisitng methods.

| Aspect | AutoML / Surrogate HPO | Existing RAG HPO | Our Work |
|--------|-------------------------|------------------|----------|
| Goal | Optimize black-box hyperparameters | Find high-performing RAG configurations for a fixed corpus | Learn the RAG configuration–performance landscape |
| Surrogate role | Guide optimization | Guide search | Enable prediction, interpretation, transfer, and optimization |
| Configuration space | Model/training hyperparameters | RAG pipeline parameters for a single corpus | Multi-stage RAG pipeline with heterogeneous interactions |
| Domain modelling | Fixed dataset | Fixed corpus | Corpus-aware modeling using corpus-level descriptors |
| Interpretability | Typically not a goal | Limited | SHAP-IQ analysis of higher-order parameter interactions |
| Knowledge transfer | Rare | Typically dataset-specific | Cross-domain surrogate transfer with minimal target-domain data |

Our contribution is therefore not a new surrogate optimization algorithm, but the observation that RAG configuration spaces constitute a learnable object. We show that surrogate models can capture the structure of these spaces, support cross-domain transfer, provide interpretable insights into parameter interactions, and enable efficient optimization of computationally expensive RAG pipelines.

## 4. Comparison of our optimization method with random search

Following the Reviewer's recommendation, we perform additional experiments thta compare our proposed approach with random search. For this, we split the original 648 configurations with already known real RAG metrics into 148 'initial' configurations and 500 'search space' configurations. We compared our method against a random search baseline under an equal evaluation budget of 10 real RAG evaluations.

For random search, 10 configurations were sampled uniformly without replacement from the 500 candidate configurations, and the best-performing configuration was selected based on the true objective value. The experiment was repeated 10 times using random seeds 0–9. 

For the proposed approach, the surrogate model was first trained on the 148 available configurations (with predictive performance estimated by five-fold cross-validation) and then retrained on the full dataset. Optuna with the TPESampler performed 1000 inexpensive surrogate evaluations by maximizing the predicted sum of Golden DOI Recall and Scientific Fact Recall. After removing duplicate suggestions, the 10 highest-ranked unique configurations were evaluated using the true RAG system metrics, and the best configuration was selected according to the true objective value. This procedure was also repeated 10 times with seeds 0–9.

### OA

| Method | Win Rate (%) | Golden DOI Recall | Scientific Fact Recall |
|--------|-------------:|------------------:|-----------------------:|
| Random Search | 30 | 0.86 ± 0.07 | 0.84 ± 0.09 |
| Our Optuna Method | **100** | **0.91 ± 0.00** | **0.90 ± 0.00** |

### MC

| Method | Win Rate (%) | Golden DOI Recall | Scientific Fact Recall |
|--------|-------------:|------------------:|-----------------------:|
| Random Search | 30 | 0.83 ± 0.05 | 0.83 ± 0.15 |
| Our Optuna Method | **100** | **0.89 ± 0.00** | **0.94 ± 0.00** |

### NZ

| Method | Win Rate (%) | Golden DOI Recall | Scientific Fact Recall |
|--------|-------------:|------------------:|-----------------------:|
| Random Search | 10 | 0.85 ± 0.04 | 0.89 ± 0.12 |
| Our Optuna Method | **100** | **0.90 ± 0.00** | **0.98 ± 0.00** |

Across all three domains, random search achieved substantially lower win rates (10–30%) and consistently inferior target metrics compared with our surrogate-guided optimization. In contrast, our method successfully identified configurations that improved the target objective in all optimization runs, demonstrating that the surrogate model provides a much more reliable search strategy than uninformed sampling. We will include these results in the camera-ready version of the manuscript. 

## 5. Human study of the answers produced by the optimized configuration compared to baseline configuration

We performed a qualitative expert inspection of some inetersting QA examples where the optimized configurations changed the generated answers. This analysis confirmed that the improvements reflected by the automatic metrics correspond to meaningful improvements in scientific QA quality. Even in cases where the answer remained incorrect, the optimized configuration often moved substantially closer to the reference answer by identifying the correct candidate antibiotic while failing only on the associated numerical value, indicating a partial improvement in reasoning and retrieval quality.


| # | Question (short) | Ground Truth | Baseline RAG | Optimized RAG | Result |
|---|------------------|--------------|--------------|---------------|--------|
| 1 | MIC of tedizolid against *M. tuberculosis* | **0.25 μg/mL** | **0.5 mg/L** ❌ Incorrect value | **0.25 mg/L** ✅ Correct value (unit mismatch) | Optimized RAG retrieved the correct MIC, whereas the baseline returned an incorrect value. |
| 2 | Tedizolid non-inferiority 95% CI | **−2.0 to 6.5** | **−10% to 10%** ❌ Incorrect confidence interval | **95% CI −2.0 to 6.5** ✅ Correct | Optimized RAG identified the correct confidence interval; the baseline confused it with the study design criterion. |
| 3 | Oxazolidinone in Gao et al. (2023) crossover study | **Delpazolid** | **Contezolid** ❌ | **Delpazolid** ✅ | Optimized RAG correctly identified the investigated compound, while the baseline confused it with another oxazolidinone. |
| 4 | Antibiotic with Ki = 182.26 μM and MIC = 2 μg/mL | **Linezolid** | Could not identify the antibiotic ❌ | **Contezolid** ❌ | Neither system produced the correct answer. The optimized configuration retrieved a plausible antibiotic that is quite close to linezolid. |
| 5 | Compare MIC values of vancomycin and linezolid | **Both are 2 μg/mL** | Vancomycin **>256 μg/mL** ❌ | Vancomycin **1 μg/mL**, linezolid **2 μg/mL** ❌ | Both systems failed. The optimized configuration retrieved a closer value for vancomycin. |