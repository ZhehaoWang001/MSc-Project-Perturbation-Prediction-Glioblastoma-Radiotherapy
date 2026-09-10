# Attention extraction

Use `run_attention.py` as the single entry point for either checkpoint family.
It selects a preserved model-specific backend and forwards all remaining
arguments unchanged. Both backend sources are copied byte-for-byte from the
reviewed project sources: Myriad's `Tutorial_Attention_GRN_full.py` for scGPT,
and `Tutorial_Attention_GRN_scGenePT_BioAll.py` for scGenePT. Their model loading,
preprocessing, sampling, attention statistic and plotting code are unchanged.
The launcher does not make the two implementations identical.

`--help` and `--dry-run` use only the Python standard library. Normal execution
uses the active Python interpreter in a subprocess and keeps the caller's
working directory. Choose the matching captured Myriad environment. This
packaging change was checked with Python syntax parsing, shell syntax checking,
and launcher help/dry-run calls; no checkpoint was loaded and no model or job was
run as part of publication preparation.

## scGPT checkpoint

The scGPT backend consumes the external raw H5AD and mouse-to-human mapping,
plus the compatible scGPT vocabulary/configuration. The model directory must
contain `best_model.pt`. Its original defaults are relative to its own script;
after packaging that script lives under `_backends/`. Supply **all five paths
explicitly**, preferably as absolute paths, as in this example:

```bash
python attention_extraction/run_attention.py --model-family scgpt \
  --model-dir /path/to/dev_perturb_adamson-Jun28-15-05 \
  --data-dir /path/to/scGPT_human \
  --data-file /path/to/private/rna_malignant_counts.h5ad \
  --mouse2human-pkl /path/to/private/mouse2human.pkl \
  --output-base /path/to/private/results/scgpt_attention \
  --n-cells 3000 --n-hvg 3000 --batch-size 1 --layer-idx 11 \
  --timepoints 24h 1week
```

`--n-cells` is the **total** cell sample for each control-versus-irradiated
timepoint comparison; its default is 3,000. Use 0 for all cells. The backend
also writes sampled-cell metadata, which belongs with the private run outputs
and is not an uploadable figure or aggregate result table.

## scGenePT checkpoint

Clone the official scGenePT repository and obtain its upstream model/embedding
assets as described in the repository's upstream setup instructions. Place the
project's `scgenept/scgenept_gb_timepoint_train_mvm_fullretrain.py` in that clone's
root: the preserved attention backend imports its model-construction helpers.
The local/private run directory must contain `gene_names.json`,
`train_expression.npy`, `val_expression.npy`, `test_expression.npy`, and each
split's matching `<split>_metadata.csv`. These restricted inputs are not in this
repository. Keep the original feature order and matching model vocabulary.

```bash
python attention_extraction/run_attention.py --model-family scgenept \
  --scgenept_repo /path/to/scGenePT \
  --models_dir /path/to/scGenePT/models \
  --run_dir /path/to/private/scGenePT_adamson_Jul11-2155 \
  --checkpoint /path/to/scGenePT_adamson_Jul11-2155/full_retrained/final_model_full_finetuned.pt \
  --model_type scgenept_bioall_gpt_concat \
  --target_conditions IR_24h,IR_1week \
  --output_dir /path/to/private/results/scgenept_attention \
  --max_cells_per_condition 4000 --batch_size 1 --layer_idx 11 --top_n 30
```

This follows the July run's full-retrained checkpoint selection. The parent
`best_model.pt` is the separate validation-selected checkpoint. The scGenePT
sample argument means cells **per condition**, unlike the scGPT total per
comparison. The shown 4,000 matches the previously reviewed BioAll job.

## Myriad submission

Submit from the repository root, replacing the Python launcher prefix in either
example with `qsub attention_extraction/submit_attention.sh scgpt` or
`qsub attention_extraction/submit_attention.sh scgenept` and keeping every
family-specific argument. For example:

```bash
qsub attention_extraction/submit_attention.sh scgpt \
  --model-dir /path/to/dev_perturb_adamson-Jun28-15-05 \
  --data-dir /path/to/scGPT_human \
  --data-file /path/to/private/rna_malignant_counts.h5ad \
  --mouse2human-pkl /path/to/private/mouse2human.pkl \
  --output-base /path/to/private/results/scgpt_attention
```

The job's shared resource requests are six hours, one GPU, 16 GB memory and
15 GB temporary storage; adjust them for the intended run. Its environment
activation follows the reviewed scGPT/scGenePT submission environments. Override
`SCGPT_VENV` (environment directory), `SCGENEPT_ENV` (Conda name/path), or
`ATTENTION_ENTRY` (absolute launcher path) with `qsub -v NAME=value,...` when
needed. The default launcher path uses the submission working directory,
because SGE may execute a spooled copy of the job file. `MODEL_FAMILY` may also
be supplied with `qsub -v MODEL_FAMILY=scgpt` instead of a positional family.
No checkpoint/data paths are invented by the shared job; provide them explicitly.

To preview a command locally, add `--dry-run` to either Python example. A dry
run prints a shell-quoted command and does not check model compatibility or load
the input data. This folder provides extraction code and the submission job;
attention outputs are generated in the output directory supplied for each run.
