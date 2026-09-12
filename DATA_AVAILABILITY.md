# Data availability

Raw and formatted glioblastoma study data cannot be distributed in this repository.
Cell metadata, training/validation/test expression arrays, prediction archives,
original split assignments and caches are excluded. No public dataset accession
or access contact has been supplied, so none is asserted.

Selected figures, aggregate metrics, pathway statistics, gene-network summaries,
code and environment records are included. Notebook outputs and attachments have
been removed. Running the complete workflow requires authorized local inputs.
Model checkpoint metadata is included; weight availability is documented in
[checkpoint information](docs/CHECKPOINTS.md). Checkpoint weights do not substitute
for the study inputs.

The [Reactome-only companion](embedding_analysis/reactome_only/README.md) can be
rerun from four included aggregate tables and the external Reactome gene-set
file. That companion does not load the restricted expression data or a checkpoint.
