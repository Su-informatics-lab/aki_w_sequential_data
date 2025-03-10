#!/bin/sh
#SBATCH --job-name=baselines
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
. .venv/bin/activate


# af, LSTM
python train.py --disease af &

# pneumonia, LSTM
python train.py --disease pneumonia &

# pd, LSTM
python train.py --disease pd &

# pod30, LSTM
python train.py --disease pod30 &

wait
