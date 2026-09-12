# Source and preparation provenance

The project scripts were run on the UCL Myriad cluster. The selected
scGPT/scGenePT training sources and environment records were copied from the
supplied project code archive. Myriad was inspected read-only during preparation.
Main training sources and both checkpoint sets were compared by SHA-256; package pins
were compared to installed metadata. No cluster files were changed during setup.

The repository's four workflows are scGPT training, scGenePT training, attention
extraction and embedding analysis. The retained Python scripts, job scripts and
notebooks match the initial uploaded versions byte-for-byte. Documentation and
environment organization were updated separately.

The scGenePT job retains its supplied settings; the saved July BioAll
configuration is distinct.
The attention entry point delegates to preserved model-specific sources rather
than changing their numerical calculations. Packaging tests cover syntax,
launcher argument forwarding and job-shell syntax, not model execution. The
shared launcher and submission job were added for publication; the model-specific
backends preserve the cluster analysis implementations.

The authoritative embedding notebook and its full Python export retain the
same 35 code cells. The published embedding notebooks have cleared execution
outputs. Selected figures and aggregate tables are provided alongside the
embedding analysis and checkpoint metadata; numerical analyses were not rerun.
Source filenames retain historical `GRN` terminology.

The cluster working directories were `~/Scratch/project_scgpt`,
`~/Scratch/scGenePT` and `~/Scratch/grn_scgpt`. Preserved analysis files also
contain development-machine paths; these do not describe a portable Myriad
installation. The [setup guide](SETUP.md) distinguishes the published folder
layout from the directories and path settings needed for a run.

External scGPT/scGenePT code and assets should be obtained from their upstream
repositories. Their original licenses apply to those materials. The local
preparation directory retains detailed private provenance; those absolute-path
inventories and conversation review reports are not uploaded.
