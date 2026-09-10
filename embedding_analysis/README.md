# Embedding analysis

Use [the full notebook](Embedding_GRN_Perturbation_Pathway_Survival.ipynb) or
[its Python export](Embedding_GRN_Perturbation_Pathway_Survival_full.py) for the
retained scGPT checkpoint. The export preserves all 35 code cells from the
authoritative notebook. Configure its paths and authorized input data before
execution; see [setup](../docs/SETUP.md).

The workflow extracts gene-token embeddings, builds cosine neighbour graphs,
clusters exploratory gene programs, performs ORA/GSEA, compares observed and
model conditional effects, and exports figures/tables. The
[Reactome-only companion](reactome_only/) reuses the main analysis results.
Saved figures and aggregate tables are under [outputs](outputs/), with the
companion's results under [outputs/reactome_only](outputs/reactome_only/).

The historical source guide retains original machine paths as provenance.
Embedding similarity edges are descriptive learned associations rather than
experimentally established directed regulatory interactions.
