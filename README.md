# MSc Project Perturbation Prediction Glioblastoma Radiotherapy

Research code for predicting radiotherapy-associated transcriptional responses
in glioblastoma using **scGPT** and **scGenePT**, with embedding-based pathway
analysis, attention extraction, and classical-model evaluation.

## Project structure

| Folder | Contents |
|---|---|
| [scgpt](scgpt/) | scGPT training code, Myriad job, requirements and June 28 checkpoint metadata |
| [scgenept](scgenept/) | scGenePT training, configurable gene-language embeddings, Myriad job and July 11 checkpoint metadata |
| [embedding_analysis](embedding_analysis/) | Full embedding workflow, pathway enrichment, Reactome-only analysis and saved outputs |
| [attention_extraction](attention_extraction/) | One launcher for scGPT/scGenePT attention extraction and a Myriad submission script |
| [evaluation](evaluation/) | Model evaluation, classical baselines, figures and aggregate result tables |
| [preprocessing](preprocessing/) | Data-formatting notebooks and gene-name repair code |
| [environments](environments/) | Compact environment guide, exact package lists and captured Myriad diagnostics |
| [docs](docs/) | Setup, model selection, checkpoint assets and provenance |

No-MVM and ECS scGPT variants are outside this repository's selected scope.

## Getting started

1. Read [setup instructions](docs/SETUP.md).
2. Obtain the official [scGPT](https://github.com/bowang-lab/scGPT) and
   [scGenePT](https://github.com/czi-ai/scGenePT) dependencies and model assets.
3. Restore the required checkpoint files using [checkpoint information](docs/CHECKPOINTS.md).
4. Supply authorized study data locally, then configure paths in the selected
   training or analysis workflow.

[scGenePT model selection](scgenept/README.md) explains NCBI, UniProt, GO and BioAll
options. [Attention instructions](attention_extraction/README.md) describe the
shared launcher and the model-specific inputs.

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

## Credits

This work uses [scGPT](https://github.com/bowang-lab/scGPT) and
[scGenePT](https://github.com/czi-ai/scGenePT). Obtain upstream code and assets
from their official projects and retain their license notices. 
