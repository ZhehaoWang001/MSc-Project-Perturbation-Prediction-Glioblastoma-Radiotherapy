# Setup

## Dependencies

Start with the [environment overview](../environments/README.md) for a side-by-side
version comparison, Myriad module settings and captured dependency checks.
Use separate scGPT and scGenePT environments. Their exact package lists contain
[164 scGPT pins](../environments/requirements-scgpt.txt) and
[166 scGenePT pins](../environments/requirements-scgenept.txt), matching the
inspected Myriad installations. Each model's `requirements.txt` points to its
shared list. The overview also retains packaging-tool versions and explains
the separate scGenePT Conda record. These are environment records, not tested
clean-install locks; the recorded scGPT dependency conflicts remain documented.

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

Training and attention workflows use the original working-directory
layout. Set paths explicitly; do not run unchanged path defaults on a different
machine. The scGenePT training source must be copied into the upstream repository
root before the current job or BioAll attention helper imports it.

The full embedding script/notebook uses `LouvainAnalysisConfig` and top-level
`PROJECT_DIR` settings. It expects raw `rna_malignant_counts.h5ad`, the original
`mouse2human.pkl`, the scGPT checkpoint and gene names, upstream `args.json` and
`vocab.json`, and `h.all.v2026.1.Hs.symbols.gmt` plus
`c2.cp.reactome.v2026.1.Hs.symbols.gmt`. The mapping/GMT files are obtained separately.
The Reactome-only notebook additionally needs the main embedding analysis outputs.
The scGenePT attention workflow additionally requires private expression splits
and their matching metadata; see the [attention guide](../attention_extraction/README.md).
Supply the authorized input files expected by each script before running it.
