#!/bin/sh
#SBATCH --job-name=aki_bidirectional_oversample_triplet_loss
#SBATCH --partition=gpuunsafe
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --mail-user=haining.wang@montana.edu
#SBATCH --mail-type=ALL

module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/12.2.0
. .venv/bin/activate


# Variant 1: LSTM, bidirectional
python train.py --disease aki --bidirectional &

# Variant 2: LSTM, unidirectional, oversampling w/t weighted class coef
python train.py --disease aki --oversample --no_weighted_ce &

# Variant 3: LSTM, bidirectional, oversampling w/t weighted class coef
python train.py --disease aki --oversample --no_weighted_ce --bidirectional &

# Variant 4: LSTM, unidirectional, triplet loss
python train.py --disease aki --triplet_loss &

# Variant 5: LSTM, bidirectional, triplet loss
python train.py --disease aki --triplet_loss --bidirectional &

wait
