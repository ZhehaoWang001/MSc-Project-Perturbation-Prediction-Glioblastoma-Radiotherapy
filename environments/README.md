# Myriad environments

Use **separate environments for scGPT and scGenePT**. These records preserve the
package versions captured from the working Myriad installations. Start with the
comparison below; the full package lists remain available when needed.

| File | Purpose |
|---|---|
| [requirements-scgpt.txt](requirements-scgpt.txt) | Exact scGPT Python package list: 164 pins |
| [requirements-scgenept.txt](requirements-scgenept.txt) | Exact scGenePT Python package list: 166 pins |
| [scgenept-conda-packages.yml](scgenept-conda-packages.yml) | Additional scGenePT Conda layer: 60 recorded entries, including native libraries |

The `requirements.txt` in each model folder points to its shared list here.
Both full lists are unchanged from the supplied exports; no package versions
were upgraded, removed or merged between models.

## Versions at a glance

| Component | scGPT | scGenePT |
|---|---|---|
| Environment | `scgpt_venv` (venv) | `scgenept_env` (Conda) |
| Python | 3.9.10 | 3.10.18 |
| Python build compiler | GCC 4.9.2 | GCC 11.2.0 |
| pip | 25.3 | 26.1.2 |
| setuptools | 59.5.0 | 69.5.1 |
| wheel | 0.45.1 | 0.47.0 |
| scgpt package | 0.2.4 | 0.2.4 |
| PyTorch | 1.13.0 | 2.1.2+cu118 |
| torchtext | 0.14.0 | 0.16.2 |
| torch-geometric | 2.6.1 | 2.5.3 |
| cell-gears | 0.0.2 | 0.0.2 |
| NumPy | 1.22.4 | 1.26.4 |
| pandas | 2.3.3 | 2.1.4 |
| SciPy | 1.13.1 | 1.15.3 |
| Scanpy | 1.9.8 | 1.11.5 |
| AnnData | 0.8.0 | 0.11.4 |
| h5py | 3.14.0 | 3.16.0 |
| scikit-learn | 1.0.2 | 1.7.2 |
| scikit-misc | 0.1.4 | 0.3.1 |

## Using these records

1. Choose the model's environment and use the corresponding full package list
   as the version reference. Packaging-tool versions are recorded in the table
   above, separately from the 164/166 package pins.
2. Follow [project setup](../docs/SETUP.md) for the official upstream code and
   required assets. Obtain scGenePT source separately; its environment also uses
   the `scgpt` Python package.
3. On Myriad, use the module and activation commands in the corresponding job:
   [scGPT job](../scgpt/submit_myriad.sh) or
   [scGenePT job](../scgenept/submit_scgenept_gb_mvm_fullretrain.sh).
   The [shared attention job](../attention_extraction/submit_attention.sh)
   selects the matching environment using the model family.

The scGenePT YAML contains only its captured Conda layer. It must be read
alongside the Python package list and packaging-tool versions above; it does
not recreate the full environment on its own. No scGPT Conda export was supplied
because its job uses a Python virtual environment.

These are **recorded environments, not tested clean-install lockfiles**.
Environment recreation and model training were not rerun during repository
preparation. The scGPT record has known dependency conflicts listed below.
CUDA-specific wheels and native libraries also require a compatible Linux/GPU
setup; a loaded CUDA module and an installed wheel's CUDA suffix are separate
parts of the record.

## Myriad module settings

These settings come from the supplied job scripts. The two exported module-log
files were empty, so they do not establish which modules were loaded when the
package exports were taken.

| Job setting | scGPT | scGenePT |
|---|---|---|
| Compiler module | `compilers/gnu/4.9.2` | `compilers/gnu/10.2.0` |
| Python module | `python/3.9.10` | `python/miniconda3/24.3.0-0` |
| CUDA module | `cuda/11.8.0/gnu-10.2.0` | `cuda/11.8.0/gnu-10.2.0` |
| Activation | `source ~/Scratch/scgpt_tutorial_folder/scgpt_venv/bin/activate` | `conda activate scgenept_env` |

## Captured dependency checks

The supplied scGenePT `pip check` reported **no broken requirements**. The
supplied scGPT check reported seven issues. This records the original diagnostics;
consolidation did not change the working environments or resolve dependencies.

<details>
<summary>scGPT: seven recorded dependency issues</summary>

```text
scib 1.1.4 requires deprecated, which is not installed.
scib 1.1.4 requires pydot, which is not installed.
chex 0.1.90 has requirement numpy>=1.24.1, but you have numpy 1.22.4.
pyro-ppl 1.9.1 has requirement torch>=2.0, but you have torch 1.13.0.
scib 1.1.4 has requirement igraph>=0.10, but you have igraph 0.9.11.
scib 1.1.4 has requirement pandas<2, but you have pandas 2.3.3.
torchmetrics 1.8.2 has requirement torch>=2.0.0, but you have torch 1.13.0.
```

</details>

## What was consolidated

The former 16 exports are represented here by one overview, two package lists
and one Conda record. Repeated raw freezes, package descriptions, Python/pip
output, diagnostics and empty module logs no longer occupy separate files.
Original exports remain in the local source archive and earlier Git history.

The raw scGenePT freeze contained local Conda build URLs for `cached-property`,
`h5py` and `pip`. The supplied normalized list already records the first two as
`cached-property==1.5.2` and `h5py==3.16.0`; pip 26.1.2 is recorded above. Those
build-machine paths are unnecessary for readers. The older minimal YAML only
listed Python 3.10, scikit-misc and h5py; their exact versions are retained in the
full records. The complete Conda dependency block is preserved, including both
`cached-property` and `cached_property` entries as originally exported.
