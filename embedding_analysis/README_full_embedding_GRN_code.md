# Full embedding gene-association analysis code

[Embedding_GRN_Perturbation_Pathway_Survival_full.py](Embedding_GRN_Perturbation_Pathway_Survival_full.py)
is a direct export of all 35 code cells from the
[authoritative notebook](Embedding_GRN_Perturbation_Pathway_Survival.ipynb).
The export preserves the code cells and adds the notebook's explanatory
Markdown as comments. The project scripts were run on UCL Myriad.

## Preserved development paths

`/Users/zhehaowang/PycharmProjects/scgpt_test/Embedding_GRN_save/Embedding_GRN_Perturbation_Pathway_Survival.ipynb`

This is the original notebook source location. The published code retains
absolute development-directory defaults; it does not require PyCharm. For a
Myriad run, configure those paths in a separate working copy as described below.

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

## Inputs and path settings on Myriad

In the first code cell, review `PROJECT_DIR`, `OUTPUT_DIR`, `MSIGDB_DIR` and the
instantiated `LouvainAnalysisConfig`. These are Python settings, not command-line
options. Changing only the shell working directory or exporting `PROJECT_DIR`
will not redirect the embedded absolute paths.

With the original relative settings, the configured project directory needs:

```text
<configured PROJECT_DIR>/
  rna_malignant_counts.h5ad
  mouse2human.pkl
  dev_perturb_adamson-Jun28-15-05/
    best_model.pt
    gene_names.json
  save/scGPT_human/
    args.json
    vocab.json
  Embedding_GRN_save/resources/msigdb_2026_1/
    h.all.v2026.1.Hs.symbols.gmt
    c2.cp.reactome.v2026.1.Hs.symbols.gmt
```

Restore the scGPT weights using [checkpoint information](../docs/CHECKPOINTS.md),
then stage them with the matching gene-name file at the configured run path.
Obtain upstream configuration/vocabulary files and GMT collections separately.
The H5AD and mouse-to-human mapping must be supplied through authorized access.
Choose an output directory for the new run; existing files with the same names
can be replaced by a rerun.

## Execution on Myriad

Use an analysis environment with the required libraries, including GSEApy and
NetworkX with `louvain_communities`; see
[environment coverage](../environments/README.md#embedding-analysis-environment-coverage).
The supplied model environment records are not a separately validated export
of the full embedding runtime.

Inside a Myriad compute allocation, set the cache paths before starting Python.
The following uses the allocation's temporary storage instead of the preserved
macOS `/private/tmp` defaults:

```bash
export MPLCONFIGDIR="$TMPDIR/msc_embedding_mpl"
export NUMBA_CACHE_DIR="$TMPDIR/msc_embedding_numba"
mkdir -p "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR"
python /path/to/configured/run/Embedding_GRN_Perturbation_Pathway_Survival_full.py
```

Replace the example script path with the configured working copy. A dedicated
embedding batch script is not included; use the intended Myriad analysis job or
compute allocation. The preserved counterfactual inference setting is `cpu`;
allocating a GPU alone does not change that Python configuration. For notebook
execution, select the matching environment as the kernel and run cells in order.

The checkpoint inference and Louvain tuning are computationally heavier than
replotting saved tables. The gene and feature counts listed above describe the
supplied analysis, rather than fixed guarantees for another dataset.

## Interpretation

The code constructs an embedding-based gene-association network, not a causal
or directed gene regulatory network. Edges indicate learned embedding
similarity. Louvain groups are exploratory gene programs, and the negative
cosine silhouette score indicates substantial overlap between them.
