"""
LSTM/GRU Classification for AKI Prediction via Recurrent Neural Networks

This module implements a recurrent classifier (LSTM or GRU) to predict Acute Kidney Injury (AKI)
from multi-channel patient vital signs data. The pipeline includes preprocessing patient CSV files,
pooling/truncating time series to a fixed length, normalizing features based on training set statistics,
and splitting data into training and validation sets. The model architecture includes residual connections,
layer normalization, dropout, and optionally an attention mechanism on top of the recurrent layer.

Usage Examples:

    # 1. Use default LSTM (2 layers) without attention:
    $ python train.py --data_dir "time_series_data_LSTM_10_29_2024" --epochs 50

    # 2. Use LSTM with attention mechanism (2 layers):
    $ python train.py --data_dir "time_series_data_LSTM_10_29_2024" --epochs 50 --attention

    # 3. Use GRU (2 layers) without attention:
    $ python train.py --data_dir "time_series_data_LSTM_10_29_2024" --epochs 50 --gru

    # 4. Use GRU with attention mechanism (2 layers):
    $ python train.py --data_dir "time_series_data_LSTM_10_29_2024" --epochs 50 --gru --attention

Additional command-line options include specifying the pooling method, batch size, learning rate,
number of epochs, and output directory for model checkpoints. For further customization, refer to the
argument parser help message.

    +-----------------------------------------------+
    |           Raw CSV Files (per patient)         |
    |  - Each CSV contains a time series with 26 channels           |
    |  - Patient ID is inferred from the filename (e.g., "R94657_...") |
    +-----------------------------------------------+
                     │
                     │  Preprocessing:
                     │    - Add "time_idx" column (assumed 1 row/second)
                     │    - (Optional) Pool over a 60-second window using a chosen method
                     │    - Compute sequence lengths for all patients and set a cap at a given percentile
                     │         (e.g., the 90th percentile of lengths becomes the fixed length)
                     │    - Truncate sequences longer than the cap; pad shorter ones
                     │    - Impute missing values with 0
                     │    - (Optional) Normalize features (using training set statistics)
                     │    - Attach static AKI label from an Excel file
                     │
                     ▼
    +-----------------------------------------------+
    |   Combined Preprocessed Data (Parquet file)   |
    |  Filename includes cap percentile (e.g., "preprocessed_90.parquet") |
    |  Columns: ID, Acute_kidney_injury, time_idx, F1, F2, …, F26      |
    +-----------------------------------------------+
                     │
                     │  Split by patient IDs into Train & Validation sets
                     │
                     ▼
    +-----------------------------------------------+
    |  PatientTimeSeriesDataset (Custom)            |
    |  Each sample is a dict with:                  |
    |    - time_series: [fixed_length, 26] tensor     |
    |      (normalized and imputed)                 |
    |    - label: scalar (AKI status)                 |
    +-----------------------------------------------+
                     │
                     ▼
    +-----------------------------------------------+
    |     DataLoader with custom collate_fn         |
    +-----------------------------------------------+
                     │
                     ▼
    +-----------------------------------------------+
    |         Recurrent Classifier Model            |
    |  Options: LSTM/GRU, with or without attention   |
    |  Input: [batch, fixed_length, 26]  --> logits    |
    +-----------------------------------------------+
                     │
                     ▼
    +-----------------------------------------------+
    |  Standard PyTorch Training Loop with wandb    |
    +-----------------------------------------------+
"""

import os
import argparse
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from collections import Counter

##########################
# Data Preprocessing Functions
##########################
def pool_minute(df, pool_window=60, pool_method="average"):
    exclude_cols = {"ID", "Acute_kidney_injury", "time_idx"}
    feature_cols = [col for col in df.columns if col not in exclude_cols and np.issubdtype(df[col].dtype, np.number)]
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
            "time_idx": window["time_idx"].mean()
        }
        for col in feature_cols:
            valid_vals = window[col].dropna()
            if pool_method == "average":
                pooled_row[col] = np.nanmean(window[col]) if len(valid_vals) > 0 else 0.0
            elif pool_method == "max":
                pooled_row[col] = np.nanmax(window[col]) if len(valid_vals) > 0 else 0.0
            elif pool_method == "median":
                pooled_row[col] = np.nanmedian(window[col]) if len(valid_vals) > 0 else 0.0
        pooled_data.append(pooled_row)
    return pd.DataFrame(pooled_data)

def truncate_pad_series(df, fixed_length, pad_value=0):
    current_length = len(df)
    if current_length >= fixed_length:
        return df.iloc[:fixed_length].copy()
    else:
        pad_df = pd.DataFrame(pad_value, index=range(fixed_length - current_length), columns=df.columns)
        for col in ["ID", "Acute_kidney_injury"]:
            if col in df.columns:
                pad_df[col] = df.iloc[0][col]
        pad_df["time_idx"] = range(current_length, fixed_length)
        return pd.concat([df, pad_df], ignore_index=True)

def compute_length_statistics(file_list, process_mode, pool_window, pool_method, label_dict):
    lengths = []
    for f in tqdm(file_list, desc="Computing sequence lengths", ncols=80):
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARNING] Error reading {f}: {e}")
            continue
        df["time_idx"] = range(len(df))
        patient_id = os.path.basename(f).split('_')[0].strip()
        df["ID"] = patient_id
        if "Acute_kidney_injury" not in df.columns:
            if patient_id in label_dict:
                df["Acute_kidney_injury"] = int(label_dict[patient_id])
            else:
                print(f"[WARNING] Patient ID {patient_id} not found in label mapping; skipping.")
                continue
        if process_mode == "pool":
            df = pool_minute(df, pool_window=pool_window, pool_method=pool_method)
        lengths.append(len(df))
    return np.array(lengths)

##########################
# Custom Dataset
##########################
class PatientTimeSeriesDataset(Dataset):
    def __init__(self, file_list, label_dict, fixed_length, process_mode="pool", pool_window=60, pool_method="average",
                 normalize=False, feature_means=None, feature_stds=None):
        self.file_list = file_list
        self.label_dict = label_dict
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
        fname = os.path.basename(csv_path)
        patient_id = fname.split('_')[0].strip()
        df = pd.read_csv(csv_path)
        df["time_idx"] = range(len(df))
        df["ID"] = patient_id
        if patient_id not in self.label_dict:
            raise KeyError(f"Patient ID {patient_id} not found in label mapping.")
        df["Acute_kidney_injury"] = int(self.label_dict[patient_id])
        if self.process_mode == "pool":
            df = pool_minute(df, pool_window=self.pool_window, pool_method=self.pool_method)
        df = truncate_pad_series(df, fixed_length=self.fixed_length)
        df["Acute_kidney_injury"] = df["Acute_kidney_injury"].astype(np.int64)
        feature_cols = [col for col in df.columns if col not in {"ID", "Acute_kidney_injury", "time_idx"}
                        and np.issubdtype(df[col].dtype, np.number)]
        time_series = torch.tensor(df[feature_cols].values, dtype=torch.float)
        time_series = torch.nan_to_num(time_series, nan=0.0)
        if self.normalize and self.feature_means is not None and self.feature_stds is not None:
            time_series = (time_series - self.feature_means) / (self.feature_stds + 1e-8)
        label = torch.tensor(int(self.label_dict[patient_id]), dtype=torch.long)
        return {"time_series": time_series, "label": label}

##########################
# Custom Data Collator
##########################
def custom_data_collator(features):
    batch = {}
    try:
        batch["time_series"] = torch.stack([f["time_series"] for f in features])
    except Exception as e:
        print(f"[DEBUG] Error stacking 'time_series': {e}")
        batch["time_series"] = [f["time_series"] for f in features]
    batch["label"] = torch.tensor([f["label"] for f in features], dtype=torch.long)
    print(f"[DEBUG] Collated batch keys: {list(batch.keys())}")
    return batch

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
# Model Variants
##########################
# Vanilla LSTM
class AKI_LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.1):
        super(AKI_LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.residual = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
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

# LSTM with Attention
class AKI_LSTMClassifierWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.1):
        super(AKI_LSTMClassifierWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.residual = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        self.layernorm = nn.LayerNorm(hidden_size)
        self.attention = Attention(hidden_size)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        rnn_out, (h_n, _) = self.lstm(x)
        # Use attention over all time steps
        context, attn_weights = self.attention(rnn_out)
        if self.residual is not None:
            residual = self.residual(x.mean(dim=1))
            context = context + residual
        context = self.layernorm(context)
        context = self.activation(context)
        context = self.dropout(context)
        logits = self.fc(context)
        return logits, attn_weights

# Vanilla GRU
class AKI_GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.1):
        super(AKI_GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.residual = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
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

# GRU with Attention
class AKI_GRUClassifierWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.1):
        super(AKI_GRUClassifierWithAttention, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.residual = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
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

##########################
# Training Loop with Early Stopping
##########################
def train_model(model, train_loader, val_loader, device, epochs, learning_rate,
                class_weights, patience=5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=0.1)
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_model_state = None
    no_improvement_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training", ncols=80,
                          leave=False):
            time_series = batch["time_series"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(time_series)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = F.cross_entropy(logits, labels, weight=class_weights.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Validation", ncols=80,
                              leave=False):
                time_series = batch["time_series"].to(device)
                labels = batch["label"].to(device)
                outputs = model(time_series)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = F.cross_entropy(logits, labels, weight=class_weights.to(device))
                val_losses.append(loss.item())
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_val_loss = np.mean(val_losses)
        acc = accuracy_score(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except Exception:
            auc = 0.5
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels,
                                                                   all_preds,
                                                                   average='binary',
                                                                   zero_division=0)
        print(
            f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Acc = {acc:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}")
        wandb.log(
            {"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
             "accuracy": acc, "auc": auc, "f1": f1})

        # Early stopping check: if validation loss improves, save the model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_f1 = f1
            best_model_state = model.state_dict()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch} after {patience} epochs with no improvement.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

##########################
# Compute Normalization Statistics
##########################
def compute_normalization_stats(dataset):
    all_data = []
    for i in tqdm(range(len(dataset)), desc="Computing normalization stats"):
        sample = dataset[i]
        all_data.append(sample["time_series"])
    all_data = torch.stack(all_data, dim=0)
    means = all_data.mean(dim=(0, 1))
    stds = all_data.std(dim=(0, 1))
    return means, stds

##########################
# Main Function
##########################
def main(args):
    # Create a run name for wandb and checkpoint folder
    model_type = "GRU" if args.gru else "LSTM"
    attn_str = "_ATTN" if args.attention else ""
    run_name = f"{model_type}{attn_str}_lr{args.learning_rate}_ep{args.epochs}"

    wandb.init(project="AKI_LSTM", name=run_name, config=vars(args))
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    df_labels = pd.read_excel("imputed_demo_data.xlsx")
    df_labels = df_labels[["ID", "Acute_kidney_injury"]].drop_duplicates().dropna(subset=["Acute_kidney_injury"])
    label_dict = {str(x).strip(): int(y) for x, y in zip(df_labels["ID"], df_labels["Acute_kidney_injury"])}

    file_list = [os.path.join(args.data_dir, fname)
                 for fname in os.listdir(args.data_dir)
                 if fname.endswith(".csv") and fname.split('_')[0].strip() in label_dict]
    if args.debug:
        file_list = file_list[:args.max_patients]
    print(f"Found {len(file_list)} patient files.")

    preprocessed_path = f"{args.preprocessed_path.split('.')[0]}_{args.cap_percentile}.parquet"
    if os.path.exists(preprocessed_path):
        print(f"Loading preprocessed data from {preprocessed_path}...")
        all_patients_df = pd.read_parquet(preprocessed_path)
        unique_ids = all_patients_df["ID"].unique()
        print(f"Loaded preprocessed data for {len(unique_ids)} patients.")
        print("Label distribution:", dict(all_patients_df["Acute_kidney_injury"].value_counts()))
        fixed_length = int(all_patients_df.groupby("ID").size().median())
    else:
        lengths = compute_length_statistics(file_list, args.process_mode, args.pool_window, args.pool_method, label_dict)
        print(f"Sequence length statistics: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.2f}, median={np.median(lengths):.2f}")
        cap_length = int(np.percentile(lengths, args.cap_percentile))
        print(f"Using cap percentile {args.cap_percentile} => fixed sequence length = {cap_length}")
        processed_samples = []
        for f in tqdm(file_list, desc="Preprocessing patient files", ncols=80):
            fname = os.path.basename(f)
            patient_id = fname.split('_')[0].strip()
            try:
                df = pd.read_csv(f)
            except Exception as e:
                print(f"[WARNING] Error reading {f}: {e}")
                continue
            df["time_idx"] = range(len(df))
            df["ID"] = patient_id
            if patient_id not in label_dict:
                print(f"[WARNING] Patient ID {patient_id} not found in label mapping; skipping.")
                continue
            df["Acute_kidney_injury"] = int(label_dict[patient_id])
            if args.process_mode == "pool":
                df = pool_minute(df, pool_window=args.pool_window, pool_method=args.pool_method)
            df = truncate_pad_series(df, fixed_length=cap_length)
            df["Acute_kidney_injury"] = df["Acute_kidney_injury"].astype(np.int64)
            processed_samples.append(df)
        if len(processed_samples) == 0:
            raise ValueError("No patient files processed successfully.")
        all_patients_df = pd.concat(processed_samples, ignore_index=True)
        pq.write_table(pa.Table.from_pandas(all_patients_df), preprocessed_path)
        print(f"Preprocessed data saved to {preprocessed_path}.")
        unique_ids = all_patients_df["ID"].unique()
        print(f"Processed data contains {len(unique_ids)} patients.")
        print("Label distribution:", dict(all_patients_df["Acute_kidney_injury"].value_counts()))
        fixed_length = cap_length

    unique_ids = all_patients_df["ID"].unique()
    labels_for_split = [label_dict[str(pid).strip()] for pid in unique_ids]
    train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=42, stratify=labels_for_split)
    train_df = all_patients_df[all_patients_df["ID"].isin(train_ids)]
    val_df = all_patients_df[all_patients_df["ID"].isin(val_ids)]
    print("Training label distribution (from data):", dict(train_df["Acute_kidney_injury"].value_counts()))
    print("Validation label distribution (from data):", dict(val_df["Acute_kidney_injury"].value_counts()))

    train_dataset = PatientTimeSeriesDataset(
        file_list=[f for f in file_list if os.path.basename(f).split('_')[0].strip() in train_ids],
        label_dict=label_dict,
        process_mode=args.process_mode,
        pool_window=args.pool_window,
        pool_method=args.pool_method,
        fixed_length=fixed_length,
        normalize=False
    )
    val_dataset = PatientTimeSeriesDataset(
        file_list=[f for f in file_list if os.path.basename(f).split('_')[0].strip() in val_ids],
        label_dict=label_dict,
        process_mode=args.process_mode,
        pool_window=args.pool_window,
        pool_method=args.pool_method,
        fixed_length=fixed_length,
        normalize=False
    )

    norm_stats_file = os.path.join(args.output_dir, "normalization_stats.pkl")
    if os.path.exists(norm_stats_file):
        with open(norm_stats_file, "rb") as f:
            stats = pickle.load(f)
        train_means = stats["means"]
        train_stds = stats["stds"]
        print(f"[DEBUG] Loaded normalization stats from {norm_stats_file}")
    else:
        train_means, train_stds = compute_normalization_stats(train_dataset)
        print(f"[DEBUG] Computed normalization means: {train_means}")
        print(f"[DEBUG] Computed normalization stds: {train_stds}")
        stats = {"means": train_means, "stds": train_stds}
        os.makedirs(args.output_dir, exist_ok=True)
        with open(norm_stats_file, "wb") as f:
            pickle.dump(stats, f)
        print(f"[DEBUG] Saved normalization stats to {norm_stats_file}")

    train_dataset.normalize = True
    train_dataset.feature_means = train_means
    train_dataset.feature_stds = train_stds
    val_dataset.normalize = True
    val_dataset.feature_means = train_means
    val_dataset.feature_stds = train_stds

    if args.debug and len(train_dataset) > 0:
        sample = train_dataset[0]
        print("[DEBUG] Sample keys:", list(sample.keys()))
        print("[DEBUG] Sample time_series shape:", sample["time_series"].shape)
        print("[DEBUG] Sample label:", sample["label"].item())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=custom_data_collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=custom_data_collator)

    all_train_labels = [train_dataset[i]["label"].item() for i in range(len(train_dataset))]
    print("Training label distribution (dataset):", dict(Counter(all_train_labels)))
    def compute_class_weights(labels):
        unique, counts = np.unique(labels, return_counts=True)
        n_samples = len(labels)
        n_classes = len(unique)
        if n_classes < 2:
            return torch.tensor([1.0, 1.0], dtype=torch.float)
        weights = n_samples / (n_classes * counts)
        weight_dict = {int(k): w for k, w in zip(unique, counts)}
        return torch.tensor([weight_dict.get(i, 1.0) for i in range(2)], dtype=torch.float)
    class_weights = compute_class_weights(all_train_labels).to(device)
    print("Computed class weights (inverse frequency):", class_weights)

    history_length = train_dataset[0]["time_series"].shape[0]
    num_input_channels = train_dataset[0]["time_series"].shape[1]
    print(f"History length (fixed): {history_length}")
    print(f"Number of input channels (observable features): {num_input_channels}")

    # Model instantiation based on flags.
    # By default, use LSTM; if --gru then use GRU.
    # If --attention flag is passed, use the corresponding attention variant.
    if args.gru:
        if args.attention:
            model = AKI_GRUClassifierWithAttention(input_size=num_input_channels, hidden_size=128,
                                                   num_layers=2, num_classes=2, dropout=0.1).to(device)
        else:
            model = AKI_GRUClassifier(input_size=num_input_channels, hidden_size=128,
                                      num_layers=2, num_classes=2, dropout=0.1).to(device)
    else:
        if args.attention:
            model = AKI_LSTMClassifierWithAttention(input_size=num_input_channels, hidden_size=128,
                                                    num_layers=2, num_classes=2, dropout=0.1).to(device)
        else:
            model = AKI_LSTMClassifier(input_size=num_input_channels, hidden_size=128,
                                       num_layers=2, num_classes=2, dropout=0.1).to(device)

    # Append run name info to the checkpoint folder
    ckpt_folder = os.path.join(args.output_dir, run_name)
    os.makedirs(ckpt_folder, exist_ok=True)

    model = train_model(model, train_loader, val_loader, device, args.epochs, args.learning_rate, class_weights, patience=args.patience)

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            time_series = batch["time_series"].to(device)
            labels = batch["label"].to(device)
            outputs = model(time_series)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except Exception:
        auc = 0.5
    print(f"Final Evaluation: Accuracy = {acc:.4f}, AUC = {auc:.4f}")

    model_path = os.path.join(ckpt_folder, "final_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    wandb.log({"final_accuracy": acc, "final_auc": auc, "checkpoint_path": model_path})
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="time_series_data_LSTM_10_29_2024",
                        help="Directory containing patient CSV files.")
    parser.add_argument("--preprocessed_path", type=str, default="preprocessed_data.parquet",
                        help="Base path to save/load preprocessed data (will append cap percentile).")
    parser.add_argument("--process_mode", type=str, choices=["truncate", "pool", "none"], default="pool",
                        help="Preprocessing mode: 'pool' uses minute pooling; 'truncate' pads/truncates to fixed length.")
    parser.add_argument("--pool_window", type=int, default=60,
                        help="Window size (in seconds) for pooling.")
    parser.add_argument("--pool_method", type=str, choices=["average", "max", "median"], default="average",
                        help="Pooling method for minute pooling.")
    parser.add_argument("--cap_percentile", type=float, default=90,
                        help="Cap percentile to determine fixed sequence length from the distribution of sequence lengths.")
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate (default now 1e-4).")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for DataLoader.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Number of epochs with no improvement on validation loss before early stopping.")
    # Hardware & debug options
    parser.add_argument("--cuda", type=int, default=0, help="GPU device index to use.")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available.")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process fewer patients and print extra info.")
    parser.add_argument("--max_patients", type=int, default=100, help="Max patients to process in debug mode.")
    parser.add_argument("--output_dir", type=str, default="./lstm_checkpoints",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--no_save", action="store_true", help="Disable model saving.")
    # New flags for model variants
    parser.add_argument("--attention", action="store_true", help="Use attention mechanism on top of the recurrent layer.")
    parser.add_argument("--gru", action="store_true", help="Use GRU instead of LSTM.")

    args = parser.parse_args()
    main(args)
