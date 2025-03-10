#!/bin/sh
#SBATCH --job-name=aki
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


# Variant 1: LSTM, no attention, no layernorm, unidirectional, dropout 0.1, lr=1e-5, weight_decay=0
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --dropout 0.1 &

# Variant 2: LSTM, attention, no layernorm, unidirectional, dropout 0.1, lr=1e-5, weight_decay=0
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --attention --dropout 0.1 &

# Variant 3: LSTM, no attention, layernorm, unidirectional, dropout 0.1, lr=1e-5, weight_decay=0
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --layernorm --dropout 0.1 &

# Variant 4: LSTM, attention, layernorm, unidirectional, dropout 0.1, lr=1e-5, weight_decay=0
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --attention --layernorm --dropout 0.1 &

# Variant 5: LSTM, no attention, no layernorm, bidirectional, dropout 0.1, lr=1e-5, weight_decay=0
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --bidirectional --dropout 0.1 &

# Variant 6: LSTM, attention, no layernorm, bidirectional, dropout 0.1, lr=1e-5, weight_decay=0
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --attention --bidirectional --dropout 0.1 &

# Variant 7: LSTM, no attention, layernorm, bidirectional, dropout 0.1, lr=1e-5, weight_decay=0
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --layernorm --bidirectional --dropout 0.1 &

# Variant 8: LSTM, attention, layernorm, bidirectional, dropout 0.1, lr=1e-5, weight_decay=0
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --attention --layernorm --bidirectional --dropout 0.1 &

# Variant 9: GRU, no attention, no layernorm, unidirectional, dropout 0.1, lr=1e-5, weight_decay=0
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --gru --dropout 0.1 &

# Variant 10: GRU, attention, no layernorm, unidirectional, dropout 0.1, lr=1e-5, weight_decay=0
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --gru --attention --dropout 0.1 &

# Variant 11: GRU, no attention, layernorm, unidirectional, dropout 0.1, lr=1e-5, weight_decay=0
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --gru --layernorm --dropout 0.1 &

# Variant 12: GRU, attention, layernorm, unidirectional, dropout 0.1, lr=1e-5, weight_decay=0
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --gru --attention --layernorm --dropout 0.1 &

# Variant 13: GRU, no attention, no layernorm, bidirectional, dropout 0.1, lr=1e-5, weight_decay=0
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --gru --bidirectional --dropout 0.1 &

# Variant 14: GRU, attention, no layernorm, bidirectional, dropout 0.1, lr=1e-5, weight_decay=0
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --gru --attention --bidirectional --dropout 0.1 &

# Variant 15: GRU, no attention, layernorm, bidirectional, dropout 0.1, lr=1e-5, weight_decay=0
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --gru --layernorm --bidirectional --dropout 0.1 &

# Variant 16: GRU, attention, layernorm, bidirectional, dropout 0.1, lr=1e-5, weight_decay=0
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --gru --attention --layernorm --bidirectional --dropout 0.1 &

# Variant 17: LSTM, attention, layernorm, unidirectional, dropout 0.2, lr=1e-5, weight_decay=0
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --attention --layernorm --dropout 0.2 &

# Variant 18: GRU, attention, layernorm, bidirectional, dropout 0.2, lr=1e-5, weight_decay=0
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --gru --attention --layernorm --bidirectional --dropout 0.2 &

# Variant 19: LSTM, attention, layernorm, bidirectional, dropout 0.1, lr=1e-5, weight_decay=0.01
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --attention --layernorm --bidirectional --dropout 0.1 --weight_decay 0.01 &

# Variant 20: GRU, attention, layernorm, bidirectional, dropout 0.1, lr=1e-5, weight_decay 0.01
python train.py --disease aki --learning_rate 1e-5 --epochs 100 --batch_size 32 --gru --attention --layernorm --bidirectional --dropout 0.1 --weight_decay 0.01 &

wait
