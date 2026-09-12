# Reactome-only companion analysis

This folder contains the notebook for the Reactome-only companion analysis,
using the MSigDB Reactome collection (`c2.cp.reactome.v2026.1.Hs.symbols.gmt`).

The notebook reuses aggregate results from the full scGPT embedding analysis:
the gene-program membership and observed/model effect statistics. The supplied
run has a 2,439-gene universe and 393 Louvain programs. This companion recalculates:

- Reactome-only program ORA and both Reactome-specific FDR corrections;
- four Reactome-only preranked GSEA runs;
- observed--model pathway recovery statistics;
- Reactome-only program annotations and presentation figures.

## Required inputs

Four inputs are already included under [the main output folder](../outputs/):

- [Gene-program membership](../outputs/embedding_gene_program_membership.csv)
- [Observed/model program effects](../outputs/gene_program_effects_observed_vs_model.csv)
- [Gene-effect rankings](../outputs/gene_effect_rankings.csv)
- [Original mixed-library GSEA results](../outputs/pathway_gsea_all.csv)

Obtain `c2.cp.reactome.v2026.1.Hs.symbols.gmt` separately. This companion reads
the four aggregate tables and that GMT file; it does not load raw expression
data, per-cell metadata or a model checkpoint.

## Run on Myriad

Open [the notebook](Reactome_Only_Embedding_Pathway_Analysis.ipynb) using a
compatible analysis environment within a Myriad compute allocation. PyCharm is
an optional editor. In a separate working copy, set `SOURCE_OUTPUT_DIR` to the
published aggregate-table folder and `REACTOME_GMT` to the downloaded collection.
These are Python settings in the first code cell. The default `OUTPUT_DIR` is
`outputs/` beneath the notebook kernel's current working directory; choose a
separate destination for new results, then run the cells in order.

Published notebook execution outputs are cleared. Selected saved figures and
aggregate tables are in [the Reactome output folder](../outputs/reactome_only/).

Important: the program-effect FDR, Reactome program-ORA FDR, Reactome-global
ORA FDR, and Reactome GSEA pathway FDR are different quantities and should not
be compared as though they were the same test.
