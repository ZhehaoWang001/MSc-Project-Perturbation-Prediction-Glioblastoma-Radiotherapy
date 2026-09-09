# Embedding analysis

Use `Embedding_GRN_Perturbation_Pathway_Survival.ipynb` or its full Python export.
The export preserves all 35 code cells from the authoritative notebook. Configure
its paths and authorized input data before execution; see `../docs/SETUP.md`.

The workflow extracts gene-token embeddings, builds cosine neighbour graphs,
clusters exploratory gene programs, performs ORA/GSEA, compares observed and
model conditional effects, and exports figures/tables. `reactome_only/` contains
the separate Reactome-only analysis. Existing outputs are under `outputs/`.

The historical source guide retains original machine paths as provenance.
Embedding similarity edges are descriptive learned associations rather than
experimentally established directed regulatory interactions.
