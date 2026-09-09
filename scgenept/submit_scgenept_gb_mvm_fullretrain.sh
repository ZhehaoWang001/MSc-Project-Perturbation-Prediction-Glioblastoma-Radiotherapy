#!/bin/bash -l

# Myriad Sun Grid Engine job: BioAll scGenePT MVM, eight metrics,
# held-out test evaluation, then full-data retraining for GRN.

#$ -l h_rt=18:00:00
#$ -l mem=16G
#$ -l gpu=2
#$ -l tmpfs=15G

#$ -N FINMVM_scGenePT_NCBI_2GPU_32BS
#$ -cwd
#$ -j y

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Job started on $(hostname) at $(date)"

module purge
module load compilers/gnu/10.2.0
module load cuda/11.8.0/gnu-10.2.0
module load python/miniconda3/24.3.0-0

source "$UCL_CONDA_PATH/etc/profile.d/conda.sh"
conda activate scgenept_env

cd ~/Scratch/scGenePT || exit 1

echo "Working directory: $(pwd)"
echo "Python: $(which python)"

test -f ./models/pretrained/scgpt/best_model.pt || exit 2
test -f './models/gene_embeddings/NCBI+UniProt_embeddings-gpt3.5-ada.pkl' || exit 2
test -f ./models/gene_embeddings/GO_all_gene_embeddings-gpt3.5-ada-concat.pickle || exit 2
test -f ./data/adamson/perturb_processed.h5ad || exit 2

python scgenept_gb_timepoint_train_mvm_fullretrain.py \
    --h5ad_path ./data/adamson/perturb_processed.h5ad \
    --data_dir ./data \
    --data_name adamson \
    --models_dir ./models \
    --model_type scgenept_ncbi_gpt \
    --init_from pretrained \
    --target_conditions IR_24h,IR_1week \
    --epochs 60 \
    --batch_size 32 \
    --eval_batch_size 32 \
    --include_zero_gene all \
    --mvm_probability 0.0 \
    --de_loss_weight 5.0 \
    --mean_response_loss_weight 1.0 \
    --early_stop 45 \
    --log_interval 100 \
    --full_val_interval 1 \
    --schedule_interval 1 \
    --lr 1e-4 \
    --save_dir ./save_scgenept/noMVM

# Optional full-retraining arguments (add them to the command above to enable):
#   --retrain_full
#   --full_retrain_epochs 8
#   --full_retrain_lr_factor 0.5
#   --full_retrain_schedule_interval 1
#   --full_retrain_gamma 0.9
#   --full_retrain_metric_interval 1

status=$?
echo "Job finished with status ${status} at $(date)"
exit ${status}
