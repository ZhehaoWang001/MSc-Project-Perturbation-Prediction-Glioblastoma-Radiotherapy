# MSc Project Perturbation Prediction Glioblastoma Radiotherapy

Research code for predicting radiotherapy-associated transcriptional responses
in glioblastoma. The repository contains four workflows: **scGPT training**,
**scGenePT training**, **attention extraction** and **embedding analysis**.

## Project structure

| Folder | Contents |
|---|---|
| [scgpt](scgpt/) | scGPT training code, Myriad job, requirements and June 28 checkpoint metadata |
| [scgenept](scgenept/) | scGenePT training, configurable gene-language embeddings, Myriad job and July 11 checkpoint metadata |
| [attention_extraction](attention_extraction/) | One launcher for scGPT/scGenePT attention extraction and a Myriad submission script |
| [embedding_analysis](embedding_analysis/) | scGPT checkpoint embedding analysis, gene programs, pathway enrichment, Reactome-only analysis and saved outputs |

Supporting files are in [environments](environments/) (the compact Myriad
environment guide and exact package records) and [docs](docs/) (setup,
checkpoint downloads and provenance).

## Getting started

1. Read [setup instructions](docs/SETUP.md).
2. Obtain the official [scGPT](https://github.com/bowang-lab/scGPT) and
   [scGenePT](https://github.com/czi-ai/scGenePT) dependencies and model assets.
3. Restore the required checkpoint files using [checkpoint information](docs/CHECKPOINTS.md).
4. Supply authorized study data locally, then configure paths in the selected
   training or analysis workflow.


## Selected outputs

### Observed and model pathway responses

![Observed versus model pathway responses](embedding_analysis/outputs/pathway_observed_vs_model.png)

### Embedding-derived gene programs

![Gene-program response atlas](embedding_analysis/outputs/all_program_response_atlas.png)

Saved figures and aggregate tables are provided for inspection. They are existing
research outputs; repository preparation did not rerun the analyses. Embedding
similarity networks and attention summaries are descriptive model analyses.

## Data and reproducibility

Raw and formatted study datasets, cell metadata and prediction arrays are
excluded. Notebook execution outputs are cleared. See [data availability](DATA_AVAILABILITY.md).
The package records match the inspected Myriad environments; recreating those
environments and running models on a fresh system have not been tested here.

Training and model-specific analysis implementations are preserved from the
selected source files. The common attention launcher only selects an implementation.
See [provenance](docs/PROVENANCE.md) for checkpoint and software-version context.

## Credits and upstream dependencies

This repository contains project-specific training and analysis scripts that use
the official [scGPT](https://github.com/bowang-lab/scGPT) and
[scGenePT](https://github.com/czi-ai/scGenePT) projects. Clone those repositories
separately and follow their installation instructions.

The upstream scGPT and scGenePT software is MIT-licensed. Preserve the relevant
copyright and license notices when redistributing upstream code. Model weights,
gene embeddings and datasets may have additional terms specified by their
providers. This notice does not license the original code in this repository.
