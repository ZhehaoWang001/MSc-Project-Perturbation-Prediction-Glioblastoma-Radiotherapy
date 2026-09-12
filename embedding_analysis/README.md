# Embedding analysis

Use [the full notebook](Embedding_GRN_Perturbation_Pathway_Survival.ipynb) or
[its Python export](Embedding_GRN_Perturbation_Pathway_Survival_full.py) for the
retained scGPT checkpoint. The export preserves all 35 code cells from the
authoritative notebook. The project execution system was UCL Myriad. Configure
the preserved development-path defaults in a separate run copy and use a
compatible analysis environment; see [Myriad setup](../docs/SETUP.md) and the
[full analysis guide](README_full_embedding_GRN_code.md).

The workflow extracts gene-token embeddings, builds cosine neighbour graphs,
clusters exploratory gene programs, performs ORA/GSEA, compares observed and
model conditional effects, and exports figures/tables. The
[Reactome-only companion](reactome_only/) reuses the main analysis results.
Saved figures and aggregate tables are under [outputs](outputs/), with the
companion's results under [outputs/reactome_only](outputs/reactome_only/).

## Choose an entry point

| Task | Requirements |
|---|---|
| Inspect saved findings | Open the figures and aggregate tables under `outputs/` |
| Rerun the full embedding workflow | scGPT checkpoint and gene names, compatible upstream assets, authorized raw study data, mouse-to-human mapping and the Hallmark/Reactome GMT files |
| Rerun Reactome-only enrichment | Four supplied aggregate CSVs and the external Reactome GMT file; see the companion guide |

The full workflow has no dedicated batch job in this repository. Run a configured
copy within a suitable Myriad compute allocation; PyCharm is an optional editor.
The exact paths, cache locations and package requirements are described in the
full guide. Historical gene/program counts in the guides refer to the supplied
analysis outputs.

Embedding similarity edges are descriptive learned associations rather than
experimentally established directed regulatory interactions.
