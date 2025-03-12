#!/bin/sh
#SBATCH --job-name=cnn_baselines
##SBATCH --account=group-jasonclark
#SBATCH --partition=nextgen-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=220G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
. .venv/bin/activate


# Train baseline for Acute Kidney Injury (AKI)
python train_cnn.py --disease aki &

# Train baseline for Atrial Fibrillation (AF)
python train_cnn.py --disease af &

# Train baseline for Pneumonia
python train_cnn.py --disease pneumonia &

# Train baseline for Postoperative Delirium (PD)
python train_cnn.py --disease pd &

# Train baseline for POD30 Death
python train_cnn.py --disease pod30 &

wait