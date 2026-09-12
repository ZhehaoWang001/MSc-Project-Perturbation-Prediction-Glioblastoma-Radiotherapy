# scGenePT training and embedding options

The custom source is
[scgenept_gb_timepoint_train_mvm_fullretrain.py](scgenept_gb_timepoint_train_mvm_fullretrain.py);
the supplied [UCL Myriad job](submit_scgenept_gb_mvm_fullretrain.sh) preserves its
existing settings. Copy the training source into the
root of a clone of [czi-ai/scGenePT](https://github.com/czi-ai/scGenePT), because
it imports `models.scGenePT` and `utils.data_loading` from that repository.

See the [environment overview](../environments/README.md) for the recorded
Python/package versions; `requirements.txt` points to the shared scGenePT list.

## Myriad inputs and submission

The job explicitly changes to `~/Scratch/scGenePT`. Stage it and the custom
source in that upstream checkout as described in [setup](../docs/SETUP.md).
The supplied job expects:

```text
~/Scratch/scGenePT/
  scgenept_gb_timepoint_train_mvm_fullretrain.py
  submit_scgenept_gb_mvm_fullretrain.sh
  models/scGenePT.py
  utils/data_loading.py
  data/adamson/perturb_processed.h5ad
  models/pretrained/scgpt/
    vocab.json
    best_model.pt
  models/gene_embeddings/
    NCBI_gene_embeddings-gpt3.5-ada.pickle
    NCBI+UniProt_embeddings-gpt3.5-ada.pkl
    GO_all_gene_embeddings-gpt3.5-ada-concat.pickle
```

Supply the formatted glioblastoma H5AD privately and download model/embedding
assets from the official project. The current NCBI setting loads the NCBI file;
the preserved job also checks for NCBI+UniProt and GO-all files before starting,
so all three listed embedding files are needed to run this job unchanged.

```bash
cd "$HOME/Scratch/scGenePT"
qsub submit_scgenept_gb_mvm_fullretrain.sh
```

The job activates `scgenept_env` and requests 18 hours, two GPUs, `mem=16G` and
`tmpfs=15G`. It sets 60 epochs, training/evaluation batch sizes of 32, learning
rate `1e-4` and early stopping after 45 non-improving epochs. Its output root is
`./save_scgenept/noMVM`, preserving the supplied job's historical directory name.

## Gene-language embedding options

Select the gene-language representation using `--model_type`:

| Option | Representation | Model-zoo checkpoint filename |
|---|---|---|
| `scgpt` | scGPT baseline | `best_model_seed_42.pt` |
| `scgenept_ncbi_gpt` | NCBI gene descriptions | `best_model_gpt3.5_ada_rnd_seed_42.pt` |
| `scgenept_ncbi+uniprot_gpt` | NCBI + UniProt | `best_model_gpt3.5_ada_rnd_seed_42.pt` |
| `scgenept_go_c_gpt_concat` | GO cellular component | `best_model_gpt3.5_ada_rnd_seed_42_concat.pt` |
| `scgenept_go_f_gpt_concat` | GO molecular function | `best_model_gpt3.5_ada_rnd_seed_42_concat.pt` |
| `scgenept_go_p_gpt_concat` | GO biological process | `best_model_gpt3.5_ada_rnd_seed_42_concat.pt` |
| `scgenept_go_all_gpt_concat` | Concatenated GO categories | `best_model_gpt3.5_ada_rnd_seed_42_concat.pt` |
| `scgenept_bioall_gpt_concat` | NCBI + UniProt + GO-all | `best_model_gpt3.5_ada_rnd_seed_42_concat.pt` (matching directory required) |

BioAll maps to `scgenept_ncbi+uniprot_gpt_go_all_gpt_concat`. The code already
implements this alias. The mapping is separate from checkpoint initialization:

- `--init_from pretrained` uses `models/pretrained/scgpt/best_model.pt`.
- `--init_from finetuned` uses `MODEL_DIR_BY_TYPE` and `MODEL_FILE_BY_TYPE` under
  `models/finetuned/<model>/<dataset>/`.
- `--checkpoint_path` overrides the selected initialization checkpoint.
- `--init_from none` builds the architecture without loading initialization weights.

A public pre-finetuned BioAll model-zoo directory may not be available; use the
project BioAll checkpoint or pretrained initialization as appropriate. Download
the required NCBI/UniProt/GO embedding assets using the upstream README.

## Supplied job and retained July checkpoint

| Setting | Included job | Saved July 11 run |
|---|---|---|
| Gene-language representation | NCBI | BioAll |
| Masking probability | 0.0 | 0.2 |
| Full-data retraining | Optional arguments are commented out | Five epochs |

The July run's
[model configuration](checkpoints/scGenePT_adamson_Jul11-2155/full_retrained/model_config.json)
and [run arguments](checkpoints/scGenePT_adamson_Jul11-2155/full_retrained/run_args.json)
record its settings. Download the validation-selected and full-retrained weights
using the [checkpoint guide](../docs/CHECKPOINTS.md). The supplied job is a
different saved configuration; it is not presented as the exact command that
produced the July checkpoint.
