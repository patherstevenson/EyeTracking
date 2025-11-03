#!/bin/bash
#SBATCH --job-name=mpiifacegaze_beta_sweep
#SBATCH --output=logs/beta_%A_%a.out
#SBATCH --error=logs/beta_%A_%a.err
#SBATCH --array=0-11
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=gpuai
#SBATCH --qos=gpu-big

module purge
module load conda/23.11-py311
conda activate eyetheia

BETAS=(0.01 0.02 0.05 0.10 0.20 0.40 0.50 0.60 0.70 0.80 0.90 1.00)
DELTA=${BETAS[$SLURM_ARRAY_TASK_ID]}

PROJECT_DIR=$HOME/EyeTheia
CHECKPOINT_DIR=$PROJECT_DIR/checkpoints/beta_${DELTA}
LOG_DIR=$PROJECT_DIR/logs
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

echo "-----------------------------------------------"
echo " Job ID: $SLURM_JOB_ID (Task: $SLURM_ARRAY_TASK_ID)"
echo " Running SmoothL1 delta = ${DELTA}"
echo " Checkpoints -> ${CHECKPOINT_DIR}"
echo "-----------------------------------------------"

python3 src/main_train.py \
    --data-mode img \
    --img-root dataset/MPIIFaceGaze \
    --epochs 15 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --smooth-l1-delta ${DELTA} \
    --num-workers 8 \
    --prefetch-factor 2 \
    --pin-memory \
    --persistent-workers \
    --amp \
    --channels-last \
    --checkpoint-dir $CHECKPOINT_DIR \
    --save-every 1 \
    --resume auto

echo "-----------------------------------------------"
echo " Beta ${DELTA} training complete"
echo "-----------------------------------------------"
