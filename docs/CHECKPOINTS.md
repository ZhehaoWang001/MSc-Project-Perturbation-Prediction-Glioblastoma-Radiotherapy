# Checkpoint files

Model weights are available in the private [v0.1.0 release](https://github.com/ZhehaoWang001/MSc-Project-Perturbation-Prediction-Glioblastoma-Radiotherapy/releases/tag/v0.1.0). Sign in to a GitHub account with access to this repository to download them.

Two saved runs are included:

- **scGPT:** `dev_perturb_adamson-Jun28-15-05`, the selected model.
- **scGenePT:** `scGenePT_adamson_Jul11-2155`, the validation-selected model and its subsequent full-data retraining used for the BioAll analysis.

| Download | Restore to this path in your checkout | Bytes |
|---|---|---:|
| [scgpt_best_model.pt](https://github.com/ZhehaoWang001/MSc-Project-Perturbation-Prediction-Glioblastoma-Radiotherapy/releases/download/v0.1.0/scgpt_best_model.pt) | `scgpt/checkpoints/dev_perturb_adamson-Jun28-15-05/best_model.pt` | 207,509,431 |
| [scgenept_best_model.pt](https://github.com/ZhehaoWang001/MSc-Project-Perturbation-Prediction-Glioblastoma-Radiotherapy/releases/download/v0.1.0/scgenept_best_model.pt) | `scgenept/checkpoints/scGenePT_adamson_Jul11-2155/best_model.pt` | 1,703,457,204 |
| [scgenept_full_retrained.pt](https://github.com/ZhehaoWang001/MSc-Project-Perturbation-Prediction-Glioblastoma-Radiotherapy/releases/download/v0.1.0/scgenept_full_retrained.pt) | `scgenept/checkpoints/scGenePT_adamson_Jul11-2155/full_retrained/final_model_full_finetuned.pt` | 1,703,457,876 |

The model weights are stored as release assets. Configuration, gene-name and selected result files are in the corresponding repository folders. [checkpoint_assets.json](checkpoint_assets.json) records exact destination paths, byte counts and SHA-256 hashes; verify downloaded files against this manifest before use.

The restore paths above describe the GitHub checkout. For Myriad, pass those
absolute paths to the attention launcher or stage the matching files in the
analysis layout described in [setup](SETUP.md). These are project fine-tuned
weights; the training jobs use separate upstream pretrained assets by default.

Use the repository's `main` branch for the current code and guides. The automatic
source archives attached to `v0.1.0` reflect that release's original code snapshot;
download the named `.pt` assets for the checkpoint files.

For example, `shasum -a 256 scgpt_best_model.pt` should match the scGPT hash in the manifest. Rename downloaded assets to their listed destination filenames when restoring the checkpoint folders.

The full-data scGenePT model and its validation-selected predecessor have different purposes. Preserve that distinction when reporting performance. Private expression splits and metadata required for analysis are not included with either checkpoint.
