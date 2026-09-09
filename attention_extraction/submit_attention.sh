#!/bin/bash -l
# Submit from the repository root; all backend arguments follow the family.
# Example: qsub attention_extraction/submit_attention.sh scgpt --model-dir /path/to/run ...
# Alternatively, export MODEL_FAMILY using qsub -v and pass only backend options.
#$ -l h_rt=6:00:00
#$ -l mem=16G
#$ -l gpu=1
#$ -l tmpfs=15G
#$ -N MSc_attention
#$ -cwd
#$ -j y

set -eo pipefail

case "${1:-}" in
    scgpt|scgenept) MODEL_FAMILY="$1"; shift ;;
    ""|--*) MODEL_FAMILY="${MODEL_FAMILY:-}" ;;
    *) echo "First argument must be scgpt or scgenept." >&2; exit 2 ;;
esac
case "$MODEL_FAMILY" in
    scgpt|scgenept) ;;
    *) echo "Specify scgpt or scgenept as the first argument, or set MODEL_FAMILY." >&2; exit 2 ;;
esac

ATTENTION_ENTRY="${ATTENTION_ENTRY:-${SGE_O_WORKDIR:-$PWD}/attention_extraction/run_attention.py}"
if [[ ! -f "$ATTENTION_ENTRY" ]]; then
    echo "Missing launcher: $ATTENTION_ENTRY. Submit from the repository root or set ATTENTION_ENTRY." >&2
    exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}"
module purge
if [[ "$MODEL_FAMILY" == scgpt ]]; then
    module load compilers/gnu/4.9.2
    module load python/3.9.10
    module load cuda/11.8.0/gnu-10.2.0
    SCGPT_VENV="${SCGPT_VENV:-$HOME/Scratch/scgpt_tutorial_folder/scgpt_venv}"
    if [[ ! -f "$SCGPT_VENV/bin/activate" ]]; then
        echo "Missing scGPT environment: $SCGPT_VENV. Set SCGPT_VENV to its directory." >&2
        exit 2
    fi
    source "$SCGPT_VENV/bin/activate"
else
    module load compilers/gnu/10.2.0
    module load cuda/11.8.0/gnu-10.2.0
    module load python/miniconda3/24.3.0-0
    source "${UCL_CONDA_PATH:?The Myriad Miniconda module must set UCL_CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate "${SCGENEPT_ENV:-scgenept_env}"
fi

echo "Model family: $MODEL_FAMILY"
echo "Python: $(command -v python)"
echo "Working directory: $PWD"
exec python "$ATTENTION_ENTRY" --model-family "$MODEL_FAMILY" "$@"
