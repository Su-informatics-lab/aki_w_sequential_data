#!/usr/bin/env python
"""
LSTM Classification for AKI Prediction via LSTM

A time series classification model that uses a custom LSTM-based classifier to predict
Acute Kidney Injury (AKI) from multi-channel patient vital signs data.

Overall Process:
+--------------------------------------------------------------+
|                  Raw CSV Files (per patient)                 |
|  Each CSV file contains a time series with 26 channels.      |
|  Patient ID is inferred from the filename (e.g., "R94657_...") |
+--------------------------------------------------------------+
                │
                │  Preprocessing:
                │    - Add "time_idx" column (assume one row per second)
                │    - Pool the time series over a 60-second window
                │      (using the specified method) and then truncate/pad
                │      to a fixed length (--fixed_length)
                │    - Attach static AKI label using patient ID via Excel file
                │
                ▼
+--------------------------------------------------------------+
|       Combined Preprocessed Data (Parquet)                   |
| Columns: ID, Acute_kidney_injury, time_idx, F1, F2, …, F26     |
+--------------------------------------------------------------+
                │
                │  Split by patient ID into Train & Validation sets
                │
                ▼
+--------------------------------------------------------------+
|        PatientTimeSeriesDataset (custom Dataset)             |
| Each sample is a dict with:                                  |
|    - time_series: tensor of shape [fixed_length, 26]           |
|    - label: scalar (AKI label)                                 |
+--------------------------------------------------------------+
                │
                ▼
+--------------------------------------------------------------+
|       DataLoader (using custom collate function)             |
+--------------------------------------------------------------+
                │
                ▼
+--------------------------------------------------------------+
|            AKI_LSTMClassifier Model (LSTM-based)             |
|  Input: [batch, fixed_length, 26]                              |
|  Outputs logits for binary classification                    |
+--------------------------------------------------------------+
                │
                ▼
+--------------------------------------------------------------+
|          Training via Standard PyTorch Training Loop         |
+--------------------------------------------------------------+
"""

import os
import argparse
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

# --- Data Preprocessing Functions ---

def pool_minute(df, pool_window=60, pool_method="average"):
    """
    Pool over non-overlapping windows using the specified method (ignoring NaNs).
    Returns a DataFrame with pooled rows.
    """
    exclude_cols = {"ID", "Acute_kidney_injury", "time_idx"}
    feature_cols = [col for col in df.columns
                    if col not in exclude_cols and np.issubdtype(df[col].dtype, np.number)]
    pooled_data = []
    n = len(df)
    num_windows = int(np.ceil(n / pool_window))
    for i in range(num_windows):
        start = i * pool_window
        end = min((i + 1) * pool_window, n)
        window = df.iloc[start:end]
        pooled_row = {}
        pooled_row["ID"] = window.iloc[0]["ID"]
        pooled_row["Acute_kidney_injury"] = window.iloc[0]["Acute_kidney_injury"]
        pooled_row["time_idx"] = window["time_idx"].mean()
        for col in feature_cols:
            valid_vals = window[col].dropna()
            if pool_method == "average":
                pooled_row[col] = np.nanmean(window[col]) if len(valid_vals) > 0 else np.nan
            elif pool_method == "max":
                pooled_row[col] = np.nanmax(window[col]) if len(valid_vals) > 0 else np.nan
            elif pool_method == "median":
                pooled_row[col] = np.nanmedian(window[col]) if len(valid_vals) > 0 else np.nan
        pooled_data.append(pooled_row)
    return pd.DataFrame(pooled_data)

def truncate_pad_series(df, fixed_length, pad_value=0):
    """
    Truncate or pad DataFrame to exactly fixed_length rows.
    """
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

# --- Custom Dataset ---
class PatientTimeSeriesDataset(Dataset):
    """
    Custom dataset that reads individual patient CSV files, applies pooling (or truncation),
    and returns a dict with:
      - 'time_series': torch.FloatTensor of shape [fixed_length, num_features]
      - 'label': torch.LongTensor scalar representing the static AKI label
    """
    def __init__(self, file_list, label_dict, process_mode="pool", pool_window=60, pool_method="average", fixed_length=10800):
        self.file_list = file_list
        self.label_dict = label_dict
        self.process_mode = process_mode
        self.pool_window = pool_window
        self.pool_method = pool_method
        self.fixed_length = fixed_length

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
            # After pooling, ensure fixed length by truncating/padding:
            df = truncate_pad_series(df, fixed_length=self.fixed_length)
        elif self.process_mode == "truncate":
            df = truncate_pad_series(df, fixed_length=self.fixed_length)
        df["Acute_kidney_injury"] = df["Acute_kidney_injury"].astype(np.int64)
        # Extract observable features (exclude ID, label, time_idx)
        feature_cols = [col for col in df.columns if col not in {"ID", "Acute_kidney_injury", "time_idx"}
                        and np.issubdtype(df[col].dtype, np.number)]
        # Convert feature values to torch tensor
        time_series = torch.tensor(df[feature_cols].values, dtype=torch.float)
        label = torch.tensor(int(self.label_dict[patient_id]), dtype=torch.long)
        return {"time_series": time_series, "label": label}

# --- Custom Data Collator ---
def custom_data_collator(features):
    """
    Custom collate function that stacks "time_series" and "label" into batch tensors.
    """
    batch = {}
    try:
        batch["time_series"] = torch.stack([f["time_series"] for f in features])
    except Exception as e:
        print(f"[DEBUG] Error stacking 'time_series': {e}")
        batch["time_series"] = [f["time_series"] for f in features]
    batch["label"] = torch.tensor([f["label"] for f in features], dtype=torch.long)
    print(f"[DEBUG] Collated batch keys: {list(batch.keys())}")
    return batch

# --- LSTM Model Definition ---
class AKI_LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        """
        LSTM-based classifier for AKI prediction.
        Args:
            input_size (int): Number of observable channels.
            hidden_size (int): Hidden dimension.
            num_layers (int): Number of LSTM layers.
            num_classes (int): Number of output classes.
            dropout (float): Dropout probability.
        """
        super(AKI_LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.residual = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [B, T, C]
        out, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]  # [B, hidden_size]
        if self.residual is not None:
            residual = self.residual(x.mean(dim=1))
            h_last = h_last + residual
        h_last = self.activation(h_last)
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)
        return logits

# --- Training Loop ---
def train_model(model, train_loader, val_loader, device, epochs, learning_rate, class_weights):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_f1 = 0.0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            time_series = batch["time_series"].to(device)  # [B, T, C]
            labels = batch["label"].to(device)              # [B]
            optimizer.zero_grad()
            outputs = model(time_series)
            loss = F.cross_entropy(outputs, labels, weight=class_weights.to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                time_series = batch["time_series"].to(device)
                labels = batch["label"].to(device)
                outputs = model(time_series)
                loss = F.cross_entropy(outputs, labels, weight=class_weights.to(device))
                val_losses.append(loss.item())
                preds = torch.argmax(outputs, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_val_loss = np.mean(val_losses)
        acc = accuracy_score(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except Exception:
            auc = 0.5
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Acc = {acc:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}")
        wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "accuracy": acc, "auc": auc, "f1": f1})

        # Save best model based on F1
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_model_state = model.state_dict()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

# --- Main Function ---
def main(args):
    wandb.init(project="AKI_LSTM", config=vars(args))
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Load AKI labels from Excel and prepare label mapping.
    df_labels = pd.read_excel("imputed_demo_data.xlsx")
    df_labels = df_labels[["ID", "Acute_kidney_injury"]].drop_duplicates().dropna(subset=["Acute_kidney_injury"])
    label_dict = {str(x).strip(): int(y) for x, y in zip(df_labels["ID"], df_labels["Acute_kidney_injury"])}

    # Get list of CSV files with valid patient IDs.
    file_list = [
        os.path.join(args.data_dir, fname)
        for fname in os.listdir(args.data_dir)
        if fname.endswith(".csv") and fname.split('_')[0].strip() in label_dict
    ]
    if args.debug:
        file_list = file_list[:args.max_patients]
    print(f"Found {len(file_list)} patient files.")

    # Split file_list into train and validation sets by patient ID.
    all_ids = [os.path.basename(f).split('_')[0].strip() for f in file_list]
    unique_ids = list(set(all_ids))
    labels_for_split = [label_dict[pid] for pid in unique_ids]
    train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=42, stratify=labels_for_split)
    train_files = [f for f in file_list if os.path.basename(f).split('_')[0].strip() in train_ids]
    val_files = [f for f in file_list if os.path.basename(f).split('_')[0].strip() in val_ids]
    print(f"Train files: {len(train_files)}, Validation files: {len(val_files)}")

    # Create datasets.
    train_dataset = PatientTimeSeriesDataset(train_files, label_dict,
                                              process_mode=args.process_mode,
                                              pool_window=args.pool_window,
                                              pool_method=args.pool_method,
                                              fixed_length=args.fixed_length)
    val_dataset = PatientTimeSeriesDataset(val_files, label_dict,
                                            process_mode=args.process_mode,
                                            pool_window=args.pool_window,
                                            pool_method=args.pool_method,
                                            fixed_length=args.fixed_length)
    # Debug: inspect one sample.
    if args.debug and len(train_dataset) > 0:
        sample = train_dataset[0]
        print("[DEBUG] Sample keys:", list(sample.keys()))
        print("[DEBUG] Sample time_series shape:", sample["time_series"].shape)
        print("[DEBUG] Sample label:", sample["label"].item())

    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_data_collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_data_collator)

    # Compute class weights.
    all_train_labels = [train_dataset[i]["label"].item() for i in range(len(train_dataset))]
    print("Training label distribution:", dict(Counter(all_train_labels)))
    def compute_class_weights(labels):
        unique, counts = np.unique(labels, return_counts=True)
        n_samples = len(labels)
        n_classes = len(unique)
        if n_classes < 2:
            return torch.tensor([1.0, 1.0], dtype=torch.float)
        weights = n_samples / (n_classes * counts)
        weight_dict = {int(k): w for k, w in zip(unique, weights)}
        return torch.tensor([weight_dict.get(i, 1.0) for i in range(2)], dtype=torch.float)
    class_weights = compute_class_weights(all_train_labels).to(device)
    print("Computed class weights (inverse frequency):", class_weights)

    # Determine fixed sequence length (history length) from dataset; all samples now have fixed_length rows.
    history_length = train_dataset[0]["time_series"].shape[0]
    print(f"History length (number of rows per patient): {history_length}")
    num_input_channels = train_dataset[0]["time_series"].shape[1]
    print(f"Number of input channels (observable features): {num_input_channels}")

    # Create LSTM model.
    model = AKI_LSTMClassifier(input_size=num_input_channels, hidden_size=128,
                               num_layers=2, num_classes=2, dropout=0.3).to(device)

    # Train model.
    model = train_model(model, train_loader, val_loader, device, args.epochs, args.learning_rate, class_weights)

    # Final evaluation on validation set.
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            time_series = batch["time_series"].to(device)
            labels = batch["label"].to(device)
            outputs = model(time_series)
            preds = torch.argmax(outputs, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except Exception:
        auc = 0.5
    print(f"Final Evaluation: Accuracy = {acc:.4f}, AUC = {auc:.4f}")

    # Save final model.
    model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    wandb.log({"final_accuracy": acc, "final_auc": auc})
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="time_series_data_LSTM_10_29_2024",
                        help="Directory containing patient CSV files.")
    parser.add_argument("--preprocessed", type=str, default="false",
                        help="(Unused) Whether preprocessed data exists.")
    parser.add_argument("--process_mode", type=str, choices=["truncate", "pool", "none"], default="pool",
                        help="Preprocessing mode: 'pool' uses minute pooling; 'truncate' pads/truncates to fixed length.")
    parser.add_argument("--pool_window", type=int, default=60,
                        help="Window size (in seconds) for pooling.")
    parser.add_argument("--pool_method", type=str, choices=["average", "max", "median"], default="average",
                        help="Pooling method for minute pooling.")
    parser.add_argument("--fixed_length", type=int, default=10800,
                        help="Fixed length for truncation mode (after pooling).")
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for DataLoader.")
    # Hardware & debug options
    parser.add_argument("--cuda", type=int, default=0, help="GPU device index to use.")
    parser.add_argument("--no_cuda", action="store_true", help="Do not use CUDA even if available.")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process fewer patients and print extra info.")
    parser.add_argument("--max_patients", type=int, default=10, help="Max patients to process in debug mode.")
    parser.add_argument("--output_dir", type=str, default="./lstm_checkpoints",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--no_save", action="store_true", help="Disable model saving.")

    args = parser.parse_args()
    main(args)
