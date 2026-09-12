# Setup on UCL Myriad

The project scripts were run on UCL Myriad. This guide maps the four GitHub
workflows to their cluster working directories, software environments and inputs.
The commands below are instructions for a future run; documentation review did
not submit jobs or alter the existing Myriad installation.

## 1. Access and repository checkout

Use an account with access to Myriad and this private GitHub repository. Follow
[UCL's Myriad access instructions](https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/)
for login and off-campus access. Submit substantial computation to the scheduler;
the login node is for file preparation, lightweight checks and job submission.

For a new checkout on Myriad:

```bash
cd "$HOME/Scratch"
git clone https://github.com/ZhehaoWang001/MSc-Project-Perturbation-Prediction-Glioblastoma-Radiotherapy.git MSc_Project
PROJECT_REPO="$HOME/Scratch/MSc_Project"
```

If the checkout already exists, set `PROJECT_REPO` to its absolute path. The
variable is used only in the staging commands below.

## 2. Software and environments

Use the [environment guide](../environments/README.md) for the exact package
records, module commands and activation instructions. The scGPT job activates
`~/Scratch/scgpt_tutorial_folder/scgpt_venv`; the scGenePT job activates the
`scgenept_env` Conda environment. The shared attention job selects between them.

The two package lists contain 164 scGPT and 166 scGenePT pins. They record the
working installations; they have not been validated as clean-install lockfiles.
The separate scGenePT Conda record must be read alongside its Python package
list. For an existing Myriad setup, reuse the corresponding environment.

Obtain the original framework repositories separately. These commands create
new directories; reuse an existing checkout when it is already present:

```bash
git clone https://github.com/bowang-lab/scGPT.git "$HOME/Scratch/scGPT_upstream"
git clone https://github.com/czi-ai/scGenePT.git "$HOME/Scratch/scGenePT"
```

Both recorded model environments use the `scgpt` package at version **0.2.4**.
The inspected scGenePT checkout was commit
`5847c57e4cccd9949a7f0b5f3775d5dcae5a5eac`. A new clone follows the upstream
default branch, so check its revision when reproducing the recorded setup.
Installing current upstream source into an existing environment can replace
the recorded package version.

## 3. Working directories and file staging

The GitHub folders organize the publication. The original cluster layout was:

| Myriad directory | Role |
|---|---|
| `~/Scratch/project_scgpt` | scGPT training source, job, private data and pretrained assets |
| `~/Scratch/scGenePT` | Official scGenePT checkout, custom training source, job and assets |
| `~/Scratch/grn_scgpt` | Original attention sources and private analysis inputs/results |

Stage the selected training files alongside the corresponding run inputs. These
copy commands preserve any files already present at the destination; if a file
already exists, compare it with the selected repository version before use.

```bash
mkdir -p "$HOME/Scratch/project_scgpt"
cp -n "$PROJECT_REPO/scgpt/Tutorial_Perturbation_Prediction_v12_revised.py" "$HOME/Scratch/project_scgpt/"
cp -n "$PROJECT_REPO/scgpt/submit_myriad.sh" "$HOME/Scratch/project_scgpt/"
cp -n "$PROJECT_REPO/scgenept/scgenept_gb_timepoint_train_mvm_fullretrain.py" "$HOME/Scratch/scGenePT/"
cp -n "$PROJECT_REPO/scgenept/submit_scgenept_gb_mvm_fullretrain.sh" "$HOME/Scratch/scGenePT/"
```

Keep the package records in the GitHub checkout: each model's `requirements.txt`
refers to `../environments/`. Copying that wrapper alone into a run directory
would break its relative reference.

## 4. Required inputs and model assets

The [scGPT guide](../scgpt/README.md) and [scGenePT guide](../scgenept/README.md)
list the files expected by each unchanged training job. Both jobs use the
historical local label `adamson`; the file at
`data/adamson/perturb_processed.h5ad` must be the authorized glioblastoma study
input. Downloading the public Adamson benchmark does not reproduce this study.

Obtain scGPT whole-human weights, `args.json` and `vocab.json` from the
[official scGPT model resources](https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo).
For scGenePT, follow the [official asset instructions](https://github.com/czi-ai/scGenePT)
for `models/pretrained/scgpt/` and `models/gene_embeddings/`. The selected
gene-language representation determines which embedding files are loaded;
the preserved job also has additional file checks documented in its guide.

For attention or embedding analysis of the project's saved models, download the
[checkpoint release assets](CHECKPOINTS.md). These fine-tuned weights are
separate from the upstream pretrained weights used to initialize training.
Keep each checkpoint with its matching gene order, configuration and vocabulary.

Raw/formatted study data, expression splits and cell metadata must be supplied
through authorized local access. See [data availability](../DATA_AVAILABILITY.md)
for what is included in the repository.

## 5. Submit training and attention jobs

After staging inputs and confirming the matching environment is available:

```bash
cd "$HOME/Scratch/project_scgpt"
qsub submit_myriad.sh
```

```bash
cd "$HOME/Scratch/scGenePT"
qsub submit_scgenept_gb_mvm_fullretrain.sh
```

scGPT resolves its source, data and output paths from the submission directory
(`#$ -cwd`). The scGenePT job explicitly changes to `~/Scratch/scGenePT` before
running its Python source. Its selected settings and the saved July BioAll run
are documented separately in the [scGenePT guide](../scgenept/README.md).

For attention, use the [shared submission examples](../attention_extraction/README.md)
from the GitHub checkout root. The packaged launcher selects the preserved
scGPT or scGenePT implementation and accepts explicit checkpoint/data paths.
It can read the existing private inputs under `~/Scratch/grn_scgpt` without
moving that directory into the repository.

The supplied job directives request:

| Job | Wall time (`h_rt`) | GPUs (`gpu`) | Memory request (`mem`) | Temporary storage (`tmpfs`) |
|---|---|---:|---|---|
| scGPT training | 16 hours | 2 | 16G | 15G |
| scGenePT training | 18 hours | 2 | 16G | 15G |
| Shared attention job | 6 hours | 1 | 16G | 15G |

These values come from the active `#$` directives. UCL defines `mem` as memory
per requested CPU slot, not GPU memory; neither training job specifies a `-pe`
parallel environment. See the [UCL scheduler guide](https://www.rc.ucl.ac.uk/docs/Experienced_Users/)
and [example jobs](https://www.rc.ucl.ac.uk/docs/Example_Jobscripts/).

Use `qstat` to monitor jobs and `qstat -j JOB_ID` to inspect a specific job.
Read the merged job log (`-j y`) and confirm the expected outputs. In particular,
the scGPT job's final shell message can appear after a Python failure, so job
completion alone is insufficient. Use separate run/data workspaces for concurrent
training: both training sources clear a `processed` cache directory at startup.

## 6. Embedding analysis on Myriad

Follow the [embedding guide](../embedding_analysis/README.md) for the full scGPT
checkpoint analysis, then the [Reactome companion](../embedding_analysis/reactome_only/README.md)
if required. The companion can also reuse four published aggregate tables
directly, together with an external Reactome GMT file.

The preserved notebook/export contains absolute development-machine paths.
Configure the Python path settings in a separate run copy for Myriad; changing
only the shell working directory will not redirect those paths. The
[full analysis guide](../embedding_analysis/README_full_embedding_GRN_code.md)
lists the required layout and settings. A dedicated embedding batch job and a
separately validated embedding environment export are not included. Execute the
configured analysis within a suitable Myriad compute allocation, following
[UCL's interactive-job guidance](https://www.rc.ucl.ac.uk/docs/Interactive_Jobs/)
or the job wrapper used for the intended analysis run.
