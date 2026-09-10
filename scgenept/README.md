# scGenePT training and embedding options

The custom source is `scgenept_gb_timepoint_train_mvm_fullretrain.py`; the supplied
Myriad job preserves its existing settings. Copy the training source into the
root of a clone of [czi-ai/scGenePT](https://github.com/czi-ai/scGenePT), because
it imports `models.scGenePT` and `utils.data_loading` from that repository.

See the [environment overview](../environments/README.md) for the recorded
Python/package versions; `requirements.txt` points to the shared scGenePT list.

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
| `scgenept_bioall_gpt_concat` | NCBI + UniProt + GO-all | Alias below; requires matching model assets |

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

The included job currently selects NCBI and masking probability 0.0; full-data
retraining arguments are optional/commented. The July 11 checkpoint's saved
configuration instead records BioAll, masking probability 0.2, and five full-data
retraining epochs. Its `model_config.json` and `run_args.json` describe that run.
Model selection is configurable; these are distinct saved configurations.
