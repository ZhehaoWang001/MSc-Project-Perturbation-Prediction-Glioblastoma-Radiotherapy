#!/bin/bash -l

# Batch Script for Myriad Cluster (Sun Grid Engine)

# 1. Request Resources
#$ -l h_rt=16:00:00        # Request 6 hrs runtime
#$ -l mem=16G               # Request 16GB RAM
#$ -l gpu=2                # Request 4 GPU
#$ -l tmpfs=15G            # Request 15GB temporary local storage

# 2. Job Info
#$ -N FINscgpt_perturb_2GPU_lossweight5_32BS_60epoch_1r1e-4_Sched1_dummy       
# Job name
#$ -cwd                    # Run in current directory
#$ -j y                    # Merge stdout and stderr

# Help PyTorch handle memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 3. Set up Environment
echo "Job started on $(hostname) at $(date)"

# Load Modules - MATCHING YOUR EXISTING ENVIRONMENT
module purge
module load compilers/gnu/4.9.2
module load python/3.9.10
module load cuda/11.8.0/gnu-10.2.0

# Activate the virtual environment
# This points to the 'scgpt_venv' inside your current folder
source ~/Scratch/scgpt_tutorial_folder/scgpt_venv/bin/activate


# <--- RECOMMENDED ADDITION: Fix memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Define your script with the full path
PYTHON_SCRIPT="Tutorial_Perturbation_Prediction_v12_revised.py"

# Use 'basename' to extract just the file name for the log
SCRIPT_NAME=$(basename $PYTHON_SCRIPT)

echo "=========================================================="
echo "Starting Myriad Job ID: $JOB_ID"
echo "Executing Python script: $SCRIPT_NAME"
echo "=========================================================="
# 4. Run the Script
echo "Python: $(which python)"
echo "Running training..."

# Run with optimized settings for the cluster
python $PYTHON_SCRIPT \
    --data_name adamson \
    --epochs 60 \
    --batch_size 32 \
    --eval_batch_size 32 \
    --early_stop 32 \
    --schedule_interval 1 \
    --lr 1e-4 \
    --save_dir ./save \
    --load_model ./save/scGPT_human
   
echo "Job finished at $(date)"
