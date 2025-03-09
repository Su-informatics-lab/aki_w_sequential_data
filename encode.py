"""
encode.py

This script loads a trained recurrent model checkpoint (LSTM or GRU, with or without
attention) and encodes the sequential data from patient CSV files into fixed-length
feature vectors. The resulting data (patient ID and the corresponding vector) is
saved as a Parquet file for use in downstream pipelines.

Usage Examples:
    # Encode using a trained LSTM without attention:
    $ python encode.py --data_dir "time_series_data_LSTM_10_29_2024" --ckpt_path "./lstm_checkpoints/LSTM_lr0.0001_ep100/final_model.pt" --preprocessed_path "preprocessed_data.parquet"

    # Encode using a trained LSTM with attention:
    $ python encode.py --data_dir "time_series_data_LSTM_10_29_2024" --ckpt_path "./lstm_checkpoints/LSTM_ATTN_lr0.0001_ep100/final_model.pt" --preprocessed_path "preprocessed_data.parquet" --attention

    # Encode using a trained GRU without attention:
    $ python encode.py --data_dir "time_series_data_LSTM_10_29_2024" --ckpt_path "./lstm_checkpoints/GRU_lr0.0001_ep100/final_model.pt" --preprocessed_path "preprocessed_data.parquet" --gru

    # Encode using a trained GRU with attention:
    $ python encode.py --data_dir "time_series_data_LSTM_10_29_2024" --ckpt_path "./lstm_checkpoints/GRU_ATTN_lr0.0001_ep100/final_model.pt" --preprocessed_path "preprocessed_data.parquet" --gru --attention

For further details on command-line options, run:
    python encode.py --help
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


##########################
# Attention Module
##########################
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, rnn_outputs):
        # rnn_outputs: [batch, time, hidden_size]
        attn_scores = self.attn(rnn_outputs)  # [batch, time, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch, time, 1]
        context = torch.sum(attn_weights * rnn_outputs, dim=1)  # [batch, hidden_size]
        return context, attn_weights


##########################
# Data Preprocessing Functions (same as in train.py)
##########################
def pool_minute(df, pool_window=60, pool_method="average"):
    exclude_cols = {"ID", "Acute_kidney_injury", "time_idx"}
    feature_cols = [
        col
        for col in df.columns
        if col not in exclude_cols and np.issubdtype(df[col].dtype, np.number)
    ]
    pooled_data = []
    n = len(df)
    num_windows = int(np.ceil(n / pool_window))
    for i in range(num_windows):
        start = i * pool_window
        end = min((i + 1) * pool_window, n)
        window = df.iloc[start:end]
        pooled_row = {
            "ID": window.iloc[0]["ID"],
            "Acute_kidney_injury": window.iloc[0]["Acute_kidney_injury"],
            "time_idx": window["time_idx"].mean(),
        }
        for col in feature_cols:
            valid_vals = window[col].dropna()
            if pool_method == "average":
                pooled_row[col] = (
                    np.nanmean(window[col]) if len(valid_vals) > 0 else 0.0
                )
            elif pool_method == "max":
                pooled_row[col] = np.nanmax(window[col]) if len(valid_vals) > 0 else 0.0
            elif pool_method == "median":
                pooled_row[col] = (
                    np.nanmedian(window[col]) if len(valid_vals) > 0 else 0.0
                )
        pooled_data.append(pooled_row)
    return pd.DataFrame(pooled_data)


def truncate_pad_series(df, fixed_length, pad_value=0):
    current_length = len(df)
    if current_length >= fixed_length:
        return df.iloc[:fixed_length].copy()
    else:
        pad_df = pd.DataFrame(
            pad_value, index=range(fixed_length - current_length), columns=df.columns
        )
        for col in ["ID", "Acute_kidney_injury"]:
            if col in df.columns:
                pad_df[col] = df.iloc[0][col]
        pad_df["time_idx"] = range(current_length, fixed_length)
        return pd.concat([df, pad_df], ignore_index=True)


##########################
# Custom Dataset for Encoding
##########################
class PatientTimeSeriesDataset(Dataset):
    def __init__(
        self,
        file_list,
        fixed_length,
        process_mode="pool",
        pool_window=60,
        pool_method="average",
        normalize=False,
        feature_means=None,
        feature_stds=None,
    ):
        self.file_list = file_list
        self.fixed_length = fixed_length
        self.process_mode = process_mode
        self.pool_window = pool_window
        self.pool_method = pool_method
        self.normalize = normalize
        self.feature_means = feature_means
        self.feature_stds = feature_stds

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        csv_path = self.file_list[idx]
        # extract patient_id from filename (assumes it's the first token)
        patient_id = os.path.basename(csv_path).split("_")[0].strip()
        df = pd.read_csv(csv_path)
        df["time_idx"] = range(len(df))
        df["ID"] = patient_id
        if self.process_mode == "pool":
            df = pool_minute(
                df, pool_window=self.pool_window, pool_method=self.pool_method
            )
        df = truncate_pad_series(df, fixed_length=self.fixed_length)
        # use only numeric features (excluding ID, label, time_idx)
        feature_cols = [
            col
            for col in df.columns
            if col not in {"ID", "Acute_kidney_injury", "time_idx"}
            and np.issubdtype(df[col].dtype, np.number)
        ]
        time_series = torch.tensor(df[feature_cols].values, dtype=torch.float)
        time_series = torch.nan_to_num(time_series, nan=0.0)
        if (
            self.normalize
            and self.feature_means is not None
            and self.feature_stds is not None
        ):
            time_series = (time_series - self.feature_means) / (
                self.feature_stds + 1e-8
            )
        return {"time_series": time_series, "patient_id": patient_id}


def custom_data_collator(features):
    batch = {}
    try:
        batch["time_series"] = torch.stack([f["time_series"] for f in features])
    except Exception as e:
        batch["time_series"] = [f["time_series"] for f in features]
    # Also collect patient IDs
    batch["patient_id"] = [f["patient_id"] for f in features]
    return batch


##########################
# Model Variants with an 'encode' Method
##########################
# Vanilla LSTM
class AKI_LSTMClassifier(nn.Module):
    def __init__(
        self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.1
    ):
        super(AKI_LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.residual = (
            nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        )
        self.layernorm = nn.LayerNorm(hidden_size)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        if self.residual is not None:
            residual = self.residual(x.mean(dim=1))
            h_last = h_last + residual
        h_last = self.layernorm(h_last)
        h_last = self.activation(h_last)
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)
        return logits

    def encode(self, x):
        out, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]
        if self.residual is not None:
            residual = self.residual(x.mean(dim=1))
            h_last = h_last + residual
        h_last = self.layernorm(h_last)
        h_last = self.activation(h_last)
        h_last = self.dropout(h_last)
        return h_last


# LSTM with Attention
class AKI_LSTMClassifierWithAttention(nn.Module):
    def __init__(
        self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.1
    ):
        super(AKI_LSTMClassifierWithAttention, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.residual = (
            nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        )
        self.layernorm = nn.LayerNorm(hidden_size)
        self.attention = Attention(hidden_size)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        rnn_out, (h_n, _) = self.lstm(x)
        context, attn_weights = self.attention(rnn_out)
        if self.residual is not None:
            residual = self.residual(x.mean(dim=1))
            context = context + residual
        context = self.layernorm(context)
        context = self.activation(context)
        context = self.dropout(context)
        logits = self.fc(context)
        return logits, attn_weights

    def encode(self, x):
        rnn_out, (h_n, _) = self.lstm(x)
        context, _ = self.attention(rnn_out)
        if self.residual is not None:
            residual = self.residual(x.mean(dim=1))
            context = context + residual
        context = self.layernorm(context)
        context = self.activation(context)
        context = self.dropout(context)
        return context


# Vanilla GRU
class AKI_GRUClassifier(nn.Module):
    def __init__(
        self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.1
    ):
        super(AKI_GRUClassifier, self).__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.residual = (
            nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        )
        self.layernorm = nn.LayerNorm(hidden_size)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, h_n = self.gru(x)
        h_last = h_n[-1]
        if self.residual is not None:
            residual = self.residual(x.mean(dim=1))
            h_last = h_last + residual
        h_last = self.layernorm(h_last)
        h_last = self.activation(h_last)
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)
        return logits

    def encode(self, x):
        out, h_n = self.gru(x)
        h_last = h_n[-1]
        if self.residual is not None:
            residual = self.residual(x.mean(dim=1))
            h_last = h_last + residual
        h_last = self.layernorm(h_last)
        h_last = self.activation(h_last)
        h_last = self.dropout(h_last)
        return h_last


# GRU with Attention
class AKI_GRUClassifierWithAttention(nn.Module):
    def __init__(
        self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.1
    ):
        super(AKI_GRUClassifierWithAttention, self).__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.residual = (
            nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        )
        self.layernorm = nn.LayerNorm(hidden_size)
        self.attention = Attention(hidden_size)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        rnn_out, h_n = self.gru(x)
        context, attn_weights = self.attention(rnn_out)
        if self.residual is not None:
            residual = self.residual(x.mean(dim=1))
            context = context + residual
        context = self.layernorm(context)
        context = self.activation(context)
        context = self.dropout(context)
        logits = self.fc(context)
        return logits, attn_weights

    def encode(self, x):
        rnn_out, h_n = self.gru(x)
        context, _ = self.attention(rnn_out)
        if self.residual is not None:
            residual = self.residual(x.mean(dim=1))
            context = context + residual
        context = self.layernorm(context)
        context = self.activation(context)
        context = self.dropout(context)
        return context


##########################
# Main Encoding Script
##########################
def main(args):
    # Build a run name (informative of architecture)
    model_type = "GRU" if args.gru else "LSTM"
    attn_str = "_ATTN" if args.attention else ""
    run_name = f"{model_type}{attn_str}_encode"

    print(f"Run name: {run_name}")

    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print(f"Using device: {device}")

    # instantiate the model (only the encoder part is needed)
    # we assume input_size is determined from a sample CSV file
    sample_file = os.path.join(args.data_dir, os.listdir(args.data_dir)[0])
    df_sample = pd.read_csv(sample_file)
    # determine feature columns (assumes numeric columns except 'ID', 'Acute_kidney_injury', 'time_idx')
    feature_cols = [
        col
        for col in df_sample.columns
        if col not in {"ID", "Acute_kidney_injury", "time_idx"}
        and np.issubdtype(df_sample[col].dtype, np.number)
    ]
    input_size = len(feature_cols)
    print(f"Input size (number of features): {input_size}")

    if args.gru:
        if args.attention:
            model = AKI_GRUClassifierWithAttention(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                num_classes=2,
                dropout=0.1,
            )
        else:
            model = AKI_GRUClassifier(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                num_classes=2,
                dropout=0.1,
            )
    else:
        if args.attention:
            model = AKI_LSTMClassifierWithAttention(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                num_classes=2,
                dropout=0.1,
            )
        else:
            model = AKI_LSTMClassifier(
                input_size=input_size,
                hidden_size=128,
                num_layers=2,
                num_classes=2,
                dropout=0.1,
            )
    model.to(device)

    # load checkpoint
    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded checkpoint from {args.ckpt_path}")

    # create dataset for encoding
    # for encoding, we use the same file_list as training
    file_list = [
        os.path.join(args.data_dir, fname)
        for fname in os.listdir(args.data_dir)
        if fname.endswith(".csv")
    ]
    # use the preprocessed length if provided, otherwise compute from a percentile
    # for simplicity, we assume a fixed_length is provided
    fixed_length = args.fixed_length
    if fixed_length is None:
        # if not provided, we could compute it from the files (here we take median length)
        lengths = []
        for f in file_list:
            df = pd.read_csv(f)
            lengths.append(len(df))
        fixed_length = int(np.median(lengths))
    print(f"Using fixed sequence length: {fixed_length}")

    dataset = PatientTimeSeriesDataset(
        file_list,
        label_dict=None,
        fixed_length=fixed_length,
        process_mode=args.process_mode,
        pool_window=args.pool_window,
        pool_method=args.pool_method,
        normalize=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_data_collator,
    )

    # Encode each sample to get the d_model vector.
    encoded_dict = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding data"):
            time_series = batch["time_series"].to(device)
            patient_ids = batch["patient_id"]
            reps = model.encode(time_series)
            reps = reps.cpu().numpy()
            for pid, rep in zip(patient_ids, reps):
                encoded_dict[pid] = rep

    # Convert to DataFrame and save as Parquet
    df_encoded = pd.DataFrame(list(encoded_dict.items()), columns=["ID", "d_model"])
    output_path = args.output_file
    df_encoded.to_parquet(output_path, index=False)
    print(f"Encoded representations saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode sequential data into fixed-length representations"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing patient CSV files.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (final_model.pt).",
    )
    parser.add_argument(
        "--preprocessed_path",
        type=str,
        default="preprocessed_data.parquet",
        help="Base path to preprocessed data (if needed).",
    )
    parser.add_argument(
        "--process_mode",
        type=str,
        choices=["truncate", "pool", "none"],
        default="pool",
        help="Preprocessing mode.",
    )
    parser.add_argument(
        "--pool_window", type=int, default=60, help="Pooling window size (in seconds)."
    )
    parser.add_argument(
        "--pool_method",
        type=str,
        choices=["average", "max", "median"],
        default="average",
        help="Pooling method.",
    )
    parser.add_argument(
        "--cap_percentile",
        type=float,
        default=90,
        help="Cap percentile for fixed sequence length.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for encoding."
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of DataLoader workers."
    )
    parser.add_argument(
        "--fixed_length",
        type=int,
        default=None,
        help="Fixed sequence length (if not provided, computed as median).",
    )
    # Model variant flags
    parser.add_argument(
        "--attention",
        action="store_true",
        help="Use attention mechanism on top of the recurrent layer.",
    )
    parser.add_argument("--gru", action="store_true", help="Use GRU instead of LSTM.")
    parser.add_argument(
        "--process_mode",
        type=str,
        choices=["truncate", "pool", "none"],
        default="pool",
        help="Preprocessing mode.",
    )
    args = parser.parse_args()
    # For encoding, we don't need label_dict, so pass an empty dict.
    # But our dataset is expecting it; we can override it in __getitem__.
    # Here we create a dummy label_dict since we only use patient IDs.
    args.label_dict = {}
    main(args)
