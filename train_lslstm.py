#!/usr/bin/env python
"""
Modified LS-LSTM Classification for Disease Prediction via Recurrent Neural Networks

This version implements the preprocessing and LS-LSTM model as described in the paper:
- Intraoperative data (28 indicators) are truncated/padded to 18,000 seconds, then additionally
  padded with 90 zeros at the start and end. They are segmented into overlapping chunks
  (segment length = 180, hop size = 120) to yield a tensor of shape (28, 180, 150).
- Missing/inconsistent intraoperative values (<=0) are imputed with the patient’s mean.
- Preoperative data (80 variables) are assumed to be present.
- Both intraoperative and preoperative features are standardized.
- The LS-LSTM model first extracts local features per segment via a bidirectional LSTM,
  then applies an inter-sequence LSTM to obtain a latent representation, which is fused with the preoperative vector.
- The training/dev split is 80/20.

Usage examples remain similar.
"""

__author__ = 'hw56@iu.edu_modified'
__version__ = '0.2'
__license__ = 'MIT'

import argparse
import os
import pickle
from collections import Counter
from torch.optim.lr_scheduler import StepLR

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

##########################
# Disease Mapping
##########################
DISEASE_MAP = {
    "aki": "Acute_kidney_injury",
    "af": "Atrial_fibrillation",
    "pneumonia": "Pneumonia",
    "pd": "Postoperative_delirium",
    "pod30": "POD30_death"
}

##########################
# Global Feature Name Lists (assumed)
##########################
# Intraoperative indicators (28)
INTRA_FEATURES = [f"intra_{i}" for i in range(1, 29)]
# Preoperative variables (80)
PREOP_FEATURES = [f"preop_{i}" for i in range(1, 81)]

##########################
# New Preprocessing Functions for LS-LSTM
##########################
def impute_intraoperative(df):
    """
    For each intraoperative indicator, replace values <= 0 with the patient’s mean for that indicator.
    Optionally, drop an indicator if >10% of its values are abnormal.
    """
    for col in INTRA_FEATURES:
        if col not in df.columns:
            continue
        # Flag abnormal values (<= 0)
        abnormal_mask = df[col] <= 0
        abnormal_rate = abnormal_mask.mean()
        if abnormal_rate > 0.10:
            # Drop the column entirely if too many abnormal values.
            df.drop(columns=[col], inplace=True)
            print(f"[DEBUG] Dropped {col} due to abnormal rate {abnormal_rate:.2f}")
        else:
            if abnormal_mask.any():
                # Impute with the patient’s mean (computed ignoring abnormal values)
                valid_mean = df.loc[~abnormal_mask, col].mean()
                df.loc[abnormal_mask, col] = valid_mean
    return df

def pad_or_truncate_series(series, target_length):
    """
    Pad with zeros (or truncate) a 2D numpy array (time_steps x features) to exactly target_length.
    """
    current_length = series.shape[0]
    if current_length >= target_length:
        return series[:target_length, :]
    else:
        pad = np.zeros((target_length - current_length, series.shape[1]), dtype=series.dtype)
        return np.concatenate([series, pad], axis=0)

def segment_intraoperative(series, seg_length=180, hop_size=120, pad_front=90, pad_back=90, num_segments=150):
    """
    Given a 2D numpy array (time_steps x features) for intraoperative data,
    first pad with pad_front zeros at the beginning and pad_back zeros at the end,
    then segment the series using a sliding window.
    Finally, output a tensor of shape (features, seg_length, num_segments).
    """
    # Pad the series (assume series is (T, features))
    series = np.pad(series, ((pad_front, pad_back), (0, 0)), mode='constant', constant_values=0)
    T, D = series.shape
    segments = []
    for i in range(num_segments):
        start = i * hop_size
        end = start + seg_length
        if end > T:
            seg = np.zeros((seg_length, D))
            available = series[start:T, :]
            seg[:available.shape[0], :] = available
        else:
            seg = series[start:end, :]
        segments.append(seg)
    segments = np.stack(segments, axis=-1)  # shape: (seg_length, D, num_segments)
    # Transpose to (D, seg_length, num_segments) to match paper notation
    return segments.transpose(1, 0, 2)

##########################
# Custom Dataset for LS-LSTM: includes both intraoperative and preoperative data.
##########################
class PatientTimeSeriesDataset(Dataset):
    def __init__(self, file_list, label_dict, disease_col, intra_target_length=18000,
                 seg_length=180, hop_size=120, pad_front=90, pad_back=90,
                 process_mode="lslstm", normalize=False, intra_feature_means=None, intra_feature_stds=None,
                 preop_feature_means=None, preop_feature_stds=None):
        """
        process_mode "lslstm" will follow the paper's protocol.
        """
        self.file_list = file_list
        self.label_dict = label_dict
        self.disease_col = disease_col
        self.intra_target_length = intra_target_length  # target length for intraoperative sequence
        self.seg_length = seg_length
        self.hop_size = hop_size
        self.pad_front = pad_front
        self.pad_back = pad_back
        self.process_mode = process_mode
        self.normalize = normalize
        self.intra_feature_means = intra_feature_means
        self.intra_feature_stds = intra_feature_stds
        self.preop_feature_means = preop_feature_means
        self.preop_feature_stds = preop_feature_stds

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        csv_path = self.file_list[idx]
        fname = os.path.basename(csv_path)
        patient_id = fname.split("_")[0].strip()
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Error reading {csv_path}: {e}")
        df["ID"] = patient_id
        if patient_id not in self.label_dict:
            raise KeyError(f"Patient ID {patient_id} not found in label mapping.")
        if self.disease_col not in df.columns:
            df[self.disease_col] = int(self.label_dict[patient_id])
        # Preprocess intraoperative data
        df = impute_intraoperative(df)
        # Select intraoperative columns that remain from INTRA_FEATURES
        intra_cols = [col for col in INTRA_FEATURES if col in df.columns]
        if len(intra_cols) == 0:
            # If no expected intraoperative columns exist, create a dummy array with zeros.
            # Expected shape: (number_of_time_steps, 28)
            dummy = np.zeros((df.shape[0], len(INTRA_FEATURES)), dtype=np.float32)
            intra_array = dummy
        else:
            intra_array = df[intra_cols].values.astype(np.float32)
            # If fewer than expected columns are present, pad with zeros to reach 28.
            if intra_array.shape[1] < len(INTRA_FEATURES):
                pad_width = len(INTRA_FEATURES) - intra_array.shape[1]
                intra_array = np.concatenate([intra_array, np.zeros(
                    (intra_array.shape[0], pad_width), dtype=np.float32)], axis=1)
        intra_array = np.nan_to_num(intra_array, nan=0.0)
        intra_array = pad_or_truncate_series(intra_array, self.intra_target_length)
        intra_tensor = segment_intraoperative(intra_array, seg_length=self.seg_length,
                                              hop_size=self.hop_size,
                                              pad_front=self.pad_front,
                                              pad_back=self.pad_back,
                                              num_segments=150)
        intra_tensor = torch.tensor(intra_tensor, dtype=torch.float)
        if self.normalize and self.intra_feature_means is not None and self.intra_feature_stds is not None:
            intra_tensor = (intra_tensor - self.intra_feature_means.view(-1, 1, 1)) / (
                        self.intra_feature_stds.view(-1, 1, 1) + 1e-8)
        # Preoperative data
        preop_df = df[[col for col in PREOP_FEATURES if col in df.columns]].copy()
        preop_array = preop_df.values.astype(np.float32)
        for j in range(preop_array.shape[1]):
            col_vals = preop_array[:, j]
            if np.isnan(col_vals).any():
                mean_val = np.nanmean(col_vals)
                col_vals[np.isnan(col_vals)] = mean_val
                preop_array[:, j] = col_vals
        preop_vector = torch.tensor(preop_array.mean(axis=0), dtype=torch.float)
        if self.normalize and self.preop_feature_means is not None and self.preop_feature_stds is not None:
            preop_vector = (preop_vector - self.preop_feature_means) / (
                        self.preop_feature_stds + 1e-8)
        label = torch.tensor(int(self.label_dict[patient_id]), dtype=torch.long)
        return {"intra_tensor": intra_tensor, "preop": preop_vector, "label": label}


##########################
# Custom Data Collator (updated to handle both intra and preop inputs)
##########################
def custom_data_collator(features):
    batch = {}
    try:
        batch["intra_tensor"] = torch.stack([f["intra_tensor"] for f in features])
        batch["preop"] = torch.stack([f["preop"] for f in features])
    except Exception as e:
        print(f"[DEBUG] Error stacking features: {e}")
    batch["label"] = torch.tensor([f["label"] for f in features], dtype=torch.long)
    return batch

##########################
# LS-LSTM Model as described in the paper
##########################
class LS_LSTMClassifier(nn.Module):
    def __init__(self, intra_feat_dim=28, seg_length=180, num_segments=150, preop_dim=80,
                 hidden_size=128, num_classes=2, dropout=0.1):
        """
        LS-LSTM Model:
         - Intra-Sequence LSTM: Processes each segment (bidirectional) to extract a local feature vector.
         - Layer Normalization on the collected segment features.
         - Inter-Sequence LSTM: Processes the sequence of segment features to output a latent intraoperative vector.
         - Fusion: Concatenate with preoperative vector.
         - FC: Final classification.
        """
        super(LS_LSTMClassifier, self).__init__()
        self.num_segments = num_segments
        self.hidden_size = hidden_size
        self.intra_lstm = nn.LSTM(input_size=intra_feat_dim, hidden_size=hidden_size,
                                  num_layers=1, batch_first=True, bidirectional=True)
        self.intra_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.inter_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,
                                  num_layers=1, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size + preop_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

    def forward(self, intra_tensor, preop):
        """
        intra_tensor: shape (batch, intra_feat_dim, seg_length, num_segments)
        preop: shape (batch, preop_dim)
        """
        B, F, L, S = intra_tensor.shape
        x = intra_tensor.permute(0, 3, 2, 1).contiguous().view(B * S, L, F)
        lstm_out, (h_n, _) = self.intra_lstm(x)
        h_n_cat = torch.cat([h_n[0], h_n[1]], dim=1)
        seg_features = self.activation(self.intra_proj(h_n_cat))
        seg_features = seg_features.view(B, S, self.hidden_size)
        seg_features = self.layernorm(seg_features)
        inter_out, (h_final, _) = self.inter_lstm(seg_features)
        intra_latent = h_final.squeeze(0)
        fused = torch.cat([intra_latent, preop], dim=1)
        fused = self.dropout(self.activation(fused))
        logits = self.fc(fused)
        return logits

    def encode(self, intra_tensor, preop):
        B, F, L, S = intra_tensor.shape
        x = intra_tensor.permute(0, 3, 2, 1).contiguous().view(B * S, L, F)
        lstm_out, (h_n, _) = self.intra_lstm(x)
        h_n_cat = torch.cat([h_n[0], h_n[1]], dim=1)
        seg_features = self.activation(self.intra_proj(h_n_cat))
        seg_features = seg_features.view(B, S, self.hidden_size)
        seg_features = self.layernorm(seg_features)
        inter_out, (h_final, _) = self.inter_lstm(seg_features)
        intra_latent = h_final.squeeze(0)
        fused = torch.cat([intra_latent, preop], dim=1)
        fused = self.dropout(self.activation(fused))
        return fused

##########################
# Training Loop with Early Stopping and Triplet Loss
##########################
from torch.nn import TripletMarginLoss

def train_model(model, train_loader, val_loader, device, epochs, learning_rate,
                class_weights, patience=5, args=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=args.weight_decay if args else 0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=args.lr_gamma)
    triplet_criterion = torch.nn.TripletMarginLoss(
        margin=args.triplet_margin if args else 1.0)
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_model_state = None
    no_improvement_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training", ncols=80,
                          leave=False):
            intra_tensor = batch["intra_tensor"].to(device)
            preop = batch["preop"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(intra_tensor, preop)
            logits = outputs
            # Use vanilla cross-entropy by default; if --weighted_ce is provided, then apply class weights.
            ce_weight = class_weights.to(device) if (
                        args and args.weighted_ce) else None
            ce_loss = F.cross_entropy(logits, labels, weight=ce_weight)

            if args and args.triplet_loss:
                embeddings = model.encode(intra_tensor, preop)
                anchor_list, pos_list, neg_list = [], [], []
                for i in range(len(labels)):
                    anchor_label = labels[i].item()
                    same_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
                    diff_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]
                    if len(same_indices) < 2 or len(diff_indices) < 1:
                        continue
                    pos_idx = np.random.choice(same_indices.cpu().numpy())
                    while pos_idx == i:
                        pos_idx = np.random.choice(same_indices.cpu().numpy())
                    neg_idx = np.random.choice(diff_indices.cpu().numpy())
                    anchor_list.append(embeddings[i].unsqueeze(0))
                    pos_list.append(embeddings[pos_idx].unsqueeze(0))
                    neg_list.append(embeddings[neg_idx].unsqueeze(0))
                if len(anchor_list) > 0:
                    anchor_t = torch.cat(anchor_list, dim=0)
                    pos_t = torch.cat(pos_list, dim=0)
                    neg_t = torch.cat(neg_list, dim=0)
                    tri_loss = triplet_criterion(anchor_t, pos_t, neg_t)
                else:
                    tri_loss = torch.tensor(0.0, device=device)
                loss = ce_loss + tri_loss
            else:
                loss = ce_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)

        # Step the learning rate scheduler at the end of the epoch
        scheduler.step()

        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Validation", ncols=80,
                              leave=False):
                intra_tensor = batch["intra_tensor"].to(device)
                preop = batch["preop"].to(device)
                labels = batch["label"].to(device)
                outputs = model(intra_tensor, preop)
                logits = outputs
                ce_loss = F.cross_entropy(logits, labels, weight=ce_weight)
                val_losses.append(ce_loss.item())
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
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "accuracy": acc,
            "auc": auc,
            "f1": f1,
            "learning_rate": scheduler.get_last_lr()[0]
        })

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
# Functions to compute normalization statistics for both intra and preop data.
##########################
def compute_normalization_stats(dataset, key):
    # key: "intra_tensor" or "preop"
    all_data = []
    for i in tqdm(range(len(dataset)), desc=f"Computing normalization stats for {key}"):
        sample = dataset[i][key]
        all_data.append(sample)
    all_data = torch.stack(all_data, dim=0)
    if key == "intra_tensor":
        means = all_data.mean(dim=(0,2,3))
        stds = all_data.std(dim=(0,2,3))
    else:
        means = all_data.mean(dim=0)
        stds = all_data.std(dim=0)
    return means, stds

##########################
# Main Function (Modified to use LS-LSTM preprocessing and model)
##########################
def main(args):
    disease_col = DISEASE_MAP[args.disease.lower()]

    # Create run name for wandb and checkpoint folder
    run_name = f"{args.disease.upper()}_LSLSTM_lr{args.learning_rate}_dr{args.dropout}"
    if args.oversample:
        run_name += "_OS"
    if args.triplet_loss:
        run_name += "_TL"
    if args.weight_decay > 0:
        run_name += f"_wd{args.weight_decay}"

    wandb.init(project=args.disease.upper(), name=run_name, config=vars(args))
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Load label file
    df_labels = pd.read_excel("imputed_demo_data.xlsx")
    df_labels = df_labels.drop_duplicates().dropna(subset=[disease_col, "ID"])
    label_dict = {str(x).strip(): int(y) for x, y in zip(df_labels["ID"], df_labels[disease_col])}

    file_list = [os.path.join(args.data_dir, fname)
                 for fname in os.listdir(args.data_dir)
                 if fname.endswith(".csv") and fname.split("_")[0].strip() in label_dict]
    if args.debug:
        file_list = file_list[: args.max_patients]
    print(f"Found {len(file_list)} patient files.")

    # For LS-LSTM, we fix intraoperative sequence length to 18000 seconds.
    intra_target_length = 18000

    # Preprocess files and save concatenated DataFrame (optional)
    processed_samples = []
    for f in tqdm(file_list, desc="Preprocessing patient files", ncols=80):
        fname = os.path.basename(f)
        patient_id = fname.split("_")[0].strip()
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARNING] Error reading {f}: {e}")
            continue
        df["ID"] = patient_id
        if patient_id not in label_dict:
            print(f"[WARNING] Patient ID {patient_id} not found in label mapping; skipping.")
            continue
        if disease_col not in df.columns:
            df[disease_col] = int(label_dict[patient_id])
        df = impute_intraoperative(df)
        processed_samples.append(df)
    if len(processed_samples) == 0:
        raise ValueError("No patient files processed successfully.")
    all_patients_df = pd.concat(processed_samples, ignore_index=True)
    preprocessed_path = f"{args.disease}_preprocessed.parquet"
    pq.write_table(pa.Table.from_pandas(all_patients_df), preprocessed_path)
    print(f"Preprocessed data saved to {preprocessed_path}.")

    # 80/20 Training/Validation split by patient ID.
    unique_ids = all_patients_df["ID"].unique()
    labels_for_split = [label_dict[str(pid).strip()] for pid in unique_ids]
    train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=42, stratify=labels_for_split)
    print(f"Training patients: {len(train_ids)}, Validation patients: {len(val_ids)}")

    train_dataset = PatientTimeSeriesDataset(
        file_list=[f for f in file_list if os.path.basename(f).split("_")[0].strip() in train_ids],
        label_dict=label_dict,
        disease_col=disease_col,
        intra_target_length=intra_target_length,
        process_mode="lslstm",
        normalize=False,
    )
    val_dataset = PatientTimeSeriesDataset(
        file_list=[f for f in file_list if os.path.basename(f).split("_")[0].strip() in val_ids],
        label_dict=label_dict,
        disease_col=disease_col,
        intra_target_length=intra_target_length,
        process_mode="lslstm",
        normalize=False,
    )

    norm_stats_file = os.path.join(args.output_dir, "normalization_stats.pkl")
    if os.path.exists(norm_stats_file):
        with open(norm_stats_file, "rb") as f:
            stats = pickle.load(f)
        intra_means = stats["intra_means"]
        intra_stds = stats["intra_stds"]
        preop_means = stats["preop_means"]
        preop_stds = stats["preop_stds"]
        print(f"[DEBUG] Loaded normalization stats from {norm_stats_file}")
    else:
        intra_means, intra_stds = compute_normalization_stats(train_dataset, key="intra_tensor")
        preop_means, preop_stds = compute_normalization_stats(train_dataset, key="preop")
        stats = {"intra_means": intra_means, "intra_stds": intra_stds,
                 "preop_means": preop_means, "preop_stds": preop_stds}
        os.makedirs(args.output_dir, exist_ok=True)
        with open(norm_stats_file, "wb") as f:
            pickle.dump(stats, f)
        print(f"[DEBUG] Saved normalization stats to {norm_stats_file}")

    train_dataset.normalize = True
    train_dataset.intra_feature_means = intra_means
    train_dataset.intra_feature_stds = intra_stds
    train_dataset.preop_feature_means = preop_means
    train_dataset.preop_feature_stds = preop_stds

    val_dataset.normalize = True
    val_dataset.intra_feature_means = intra_means
    val_dataset.intra_feature_stds = intra_stds
    val_dataset.preop_feature_means = preop_means
    val_dataset.preop_feature_stds = preop_stds

    all_train_labels = [train_dataset[i]["label"].item() for i in range(len(train_dataset))]
    if args.oversample:
        print("[DEBUG] Using oversampling (WeightedRandomSampler).")
        labels_np = np.array(all_train_labels)
        unique_labels, counts = np.unique(labels_np, return_counts=True)
        weights = np.zeros_like(labels_np, dtype=np.float32)
        for ul, c in zip(unique_labels, counts):
            weights[labels_np == ul] = 1.0 / c
        train_sampler = WeightedRandomSampler(weights, num_samples=len(labels_np), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=custom_data_collator,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=custom_data_collator,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_data_collator,
    )

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

    num_intra_channels = len(INTRA_FEATURES)
    num_preop = len(PREOP_FEATURES)
    print(f"Number of intraoperative features: {num_intra_channels}")
    print(f"Number of preoperative features: {num_preop}")

    model = LS_LSTMClassifier(
        intra_feat_dim=num_intra_channels,
        seg_length=180,
        num_segments=150,
        preop_dim=num_preop,
        hidden_size=128,
        num_classes=2,
        dropout=args.dropout,
    ).to(device)

    ckpt_folder = os.path.join(args.output_dir, run_name)
    os.makedirs(ckpt_folder, exist_ok=True)

    model = train_model(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.learning_rate,
        class_weights,
        patience=args.patience,
        args=args
    )

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            intra_tensor = batch["intra_tensor"].to(device)
            preop = batch["preop"].to(device)
            labels = batch["label"].to(device)
            outputs = model(intra_tensor, preop)
            preds = torch.argmax(outputs, dim=-1)
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
    # data parameters
    parser.add_argument("--data_dir", type=str, default="time_series_data_LSTM_10_29_2024",
                        help="Directory containing patient CSV files.")
    parser.add_argument("--process_mode", type=str, choices=["lslstm", "pool", "truncate", "none"],
                        default="lslstm", help="Preprocessing mode: 'lslstm' uses the segmentation strategy from the paper.")
    # training parameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--lr_gamma", type=float, default=0.97,
                        help="Learning rate's decay rate.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for DataLoader.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Number of epochs with no improvement on validation loss before early stopping.")
    # hardware & debug options
    parser.add_argument("--cuda", type=int, default=0, help="GPU device index to use.")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available.")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process fewer patients and print extra info.")
    parser.add_argument("--max_patients", type=int, default=100, help="Max patients to process in debug mode.")
    parser.add_argument("--output_dir", type=str, default="./lslstm_checkpoints",
                        help="Directory to save model checkpoints.")
    # Model variant flags
    parser.add_argument("--oversample", action="store_true", help="Use weighted random sampling to oversample the minority class.")
    parser.add_argument("--triplet_loss", action="store_true", help="Use triplet margin loss in addition to cross-entropy classification.")
    parser.add_argument("--triplet_margin", type=float, default=1.0, help="Margin for triplet margin loss (default: 1.0).")
    # Use vanilla cross-entropy by default; if --weighted_ce is provided, then weighted cross-entropy is used.
    parser.add_argument("--weighted_ce", action="store_true", help="Enable weighted cross-entropy loss (default: disabled).")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay to use in the optimizer (default: 0, no weight decay).")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1).")
    # Disease argument
    parser.add_argument("--disease", type=str, choices=["aki", "af", "pneumonia", "pd", "pod30"],
                        default="aki", help="Which disease label to predict. Default: aki.")

    args = parser.parse_args()
    main(args)
