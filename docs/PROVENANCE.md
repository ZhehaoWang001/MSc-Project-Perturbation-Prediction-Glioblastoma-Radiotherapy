# Source and preparation provenance

The selected scGPT/scGenePT training sources and environment records were copied
from the supplied project code archive. Myriad was inspected read-only. Main
training sources and both checkpoint sets were compared by SHA-256; package pins
were compared to installed metadata. No cluster files were changed during setup.

No-MVM and ECS scGPT variants are excluded by project scope. The scGenePT job is
retained as a configurable example; the saved July BioAll configuration is distinct.
The attention entry point delegates to preserved model-specific sources rather
than changing their numerical calculations. Packaging tests cover syntax,
launcher argument forwarding and job-shell syntax, not model execution.

The authoritative embedding notebook and its full Python export retain the
same 35 code cells. Evaluation and preprocessing notebooks have cleared outputs.
Existing figures and aggregate tables are preserved; numerical analyses were
not rerun. Source filenames retain historical `GRN` terminology.

External scGPT/scGenePT code and assets should be obtained from their upstream
repositories. Their original licenses apply to those materials. The local
preparation directory retains detailed private provenance; those absolute-path
inventories and conversation review reports are not uploaded.
