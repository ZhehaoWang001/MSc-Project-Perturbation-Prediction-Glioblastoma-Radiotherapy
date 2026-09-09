# Full embedding gene-association analysis code

`Embedding_GRN_Perturbation_Pathway_Survival_full.py` is a direct export of
all 35 code cells from the authoritative PyCharm notebook. No project-local
analysis module is imported. The export preserves every code cell exactly and
adds the notebook's explanatory Markdown as comments.

## Authoritative notebook

`/Users/zhehaowang/PycharmProjects/scgpt_test/Embedding_GRN_save/Embedding_GRN_Perturbation_Pathway_Survival.ipynb`

## Analysis covered

1. Configure paths, libraries, seeds, Louvain grids and GSEA parameters.
2. Load the fine-tuned checkpoint, vocabulary, training gene axis and
   mouse-to-human mapping.
3. Extract checkpoint gene-token embeddings, apply encoder layer
   normalisation and L2 normalisation, and retain 2,439 unique eligible genes.
4. Read `rna_malignant_counts.h5ad` and reproduce the checkpoint-aligned
   preprocessing and 2,465-feature biological axis.
5. Construct cosine Scanpy neighbour graphs, tune Louvain clustering across
   neighbourhood sizes, resolutions and seeds, and select the fixed partition.
6. Perform Hallmark/Reactome over-representation analysis of the fixed gene
   programs.
7. Score the same programs in control, 24-hour and 1-week cells.
8. Calculate observed biological-sample pseudobulk contrasts and matched
   in-silico model conditional contrasts.
9. Test program effects and create the all-program response atlas.
10. Run four preranked GSEA analyses and compare observed and model pathway
    normalised enrichment scores.
11. Generate leading-edge heatmaps, sample-level pathway diagnostics, the
    full embedding-neighbour graph and focused interferon/cell-cycle networks.
12. Audit `survival_classification` and save all transparent tables.

## Running the code

Use the `scgpt_env` interpreter in PyCharm. The notebook is the recommended
execution format because results and explanations remain grouped by section.
The Python export can also be run as one script:

```bash
/opt/anaconda3/envs/scgpt_env/bin/python Embedding_GRN_Perturbation_Pathway_Survival_full.py
```

The configured inputs and output paths are absolute. Running the complete code
again writes files to:

`/Users/zhehaowang/PycharmProjects/scgpt_test/Embedding_GRN_save/perturbation_pathway_outputs`

The checkpoint inference and Louvain tuning are computationally heavier than
plotting from the saved tables. Running the script may replace files with the
same names in that output folder.

## Interpretation

The code constructs an embedding-based gene-association network, not a causal
or directed gene regulatory network. Edges indicate learned embedding
similarity. Louvain groups are exploratory gene programs, and the negative
cosine silhouette score indicates substantial overlap between them.
