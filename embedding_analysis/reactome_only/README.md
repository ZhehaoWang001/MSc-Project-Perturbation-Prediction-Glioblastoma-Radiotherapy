# Reactome-only companion analysis

This folder contains a separate, executed analysis that uses only the
MSigDB Reactome collection (`c2.cp.reactome.v2026.1.Hs.symbols.gmt`).

The checkpoint embeddings, 2,439-gene universe, embedding-neighbour graph,
393 Louvain programs, program scores, and observed/model gene-effect rankings
are reused unchanged from `perturbation_pathway_outputs`. The notebook
recalculates:

- Reactome-only program ORA and both Reactome-specific FDR corrections;
- four Reactome-only preranked GSEA runs;
- observed--model pathway recovery statistics;
- Reactome-only program annotations and presentation figures.

Open `Reactome_Only_Embedding_Pathway_Analysis.ipynb` in PyCharm. All generated
tables, figures, the analysis manifest, and a concise findings summary are in
`outputs/`.

Important: the program-effect FDR, Reactome program-ORA FDR, Reactome-global
ORA FDR, and Reactome GSEA pathway FDR are different quantities and should not
be compared as though they were the same test.
