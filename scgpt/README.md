# scGPT training

The selected implementation is
[Tutorial_Perturbation_Prediction_v12_revised.py](Tutorial_Perturbation_Prediction_v12_revised.py),
with [submit_myriad.sh](submit_myriad.sh) for UCL Myriad. Its
[requirements.txt](requirements.txt) points to the
[shared environment record](../environments/README.md).

## Myriad inputs and submission

Stage the source and job in `~/Scratch/project_scgpt`, following
[project setup](../docs/SETUP.md). The unchanged job expects this layout:

```text
~/Scratch/project_scgpt/
  Tutorial_Perturbation_Prediction_v12_revised.py
  submit_myriad.sh
  data/adamson/perturb_processed.h5ad
  save/scGPT_human/
    args.json
    vocab.json
    best_model.pt
```

The H5AD is the private, formatted glioblastoma input with its original
GEARS-compatible annotations and differential-expression metadata. Preserve
that prepared dataset structure when staging the inputs. The files under
`save/scGPT_human/` are the upstream pretrained assets. The job activates
`~/Scratch/scgpt_tutorial_folder/scgpt_venv`, which is separate from the training
directory; its module settings are listed in the environment guide.

```bash
cd "$HOME/Scratch/project_scgpt"
qsub submit_myriad.sh
```

The job uses the submission directory for relative paths and requests 16 hours,
two GPUs, `mem=16G` and `tmpfs=15G`. Its command sets 60 epochs, training/evaluation
batch sizes of 32 and learning rate `1e-4`. New run outputs are written under
`./save/dev_perturb_adamson-<timestamp>/`. Check the Python log and expected files
after the job finishes.

## Retained checkpoint

The retained `dev_perturb_adamson-Jun28-15-05` checkpoint is an earlier saved run;
the current selected training source is not claimed to be the exact historical
revision that produced it. `adamson` is the historical cache label in the custom
workflow, not a replacement for the glioblastoma study dataset.

Download its weights using the [checkpoint guide](../docs/CHECKPOINTS.md).
The repository retains gene names, metrics and selected figures beside the
checkpoint destination. The [attention](../attention_extraction/README.md) and
[embedding](../embedding_analysis/README.md) guides describe its analysis inputs.
