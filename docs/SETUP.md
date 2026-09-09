# Setup

## Dependencies

Use separate scGPT and scGenePT environments. Their captured requirements are
under `environments/myriad/` and beside each model's training script. The 164
scGPT and 166 scGenePT pinned packages match the inspected Myriad installations.
The archived scGPT pip-check contains dependency conflicts; the archived scGenePT
check reports none. These are environment records, not tested clean-install locks.

Obtain official source separately:

```bash
git clone https://github.com/bowang-lab/scGPT.git
git clone https://github.com/czi-ai/scGenePT.git
```

Both Myriad environments use scGPT 0.2.4; the newer official source snapshot
inspected during preparation declares 0.2.5. Use the recorded runtime version
when reproducing the original environment. The inspected scGenePT checkout is
commit `5847c57e4cccd9949a7f0b5f3775d5dcae5a5eac`.

## Upstream model assets

Follow the upstream README downloads for scGPT whole-human `args.json`,
`vocab.json` and pretrained weights. scGenePT additionally needs the language
embeddings selected by `--model_type`. Its official README documents the public
S3 locations, including `s3://czi-scgenept-public/models/pretrained/scgpt` and
`models/gene_embeddings`. Use matching vocabulary and condition-token additions.

## Local inputs and paths

Training/evaluation and attention workflows use the original working-directory
layout. Set paths explicitly; do not run unchanged path defaults on a different
machine. The scGenePT training source must be copied into the upstream repository
root before the current job or BioAll attention helper imports it.

The full embedding script/notebook uses `LouvainAnalysisConfig` and top-level
`PROJECT_DIR` settings. It expects raw `rna_malignant_counts.h5ad`, the original
`mouse2human.pkl`, the scGPT checkpoint and gene names, upstream `args.json` and
`vocab.json`, and `h.all.v2026.1.Hs.symbols.gmt` plus
`c2.cp.reactome.v2026.1.Hs.symbols.gmt`. The mapping/GMT files are obtained separately.
The Reactome-only notebook additionally needs the main embedding analysis outputs.
Evaluation notebooks require private split/prediction files for full reruns.
