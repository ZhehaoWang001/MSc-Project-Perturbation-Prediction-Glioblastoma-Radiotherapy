# Checkpoint files

Two saved runs are represented:

- `scgpt/checkpoints/dev_perturb_adamson-Jun28-15-05`: selected scGPT model.
- `scgenept/checkpoints/scGenePT_adamson_Jul11-2155`: validation-selected scGenePT
  model and `full_retrained/final_model_full_finetuned.pt` for the BioAll analysis.

The model weights are excluded from ordinary Git history. See the repository's
Releases for any published checkpoint assets; availability will be recorded here
after uploads are verified. The prepared local copies remain available.
`checkpoint_assets.json` gives exact destination paths, byte counts and SHA-256
hashes. Restore each asset to its listed destination filename before use.

The full-data scGenePT model and its validation-selected predecessor have different
purposes. Preserve that distinction when reporting performance. Private expression
splits/metadata required for analysis are not included with either checkpoint.
