#!/usr/bin/env python
"""
Improved LS-LSTM Classification for Disease Prediction via Recurrent Neural Networks

This version implements the preprocessing and LS-LSTM model as described in the paper:
- Intraoperative data (28 indicators) are truncated/padded to 18,000 seconds, then additionally
  padded with 90 zeros at the start and end. They are segmented into overlapping chunks
  (segment length = 180, hop size = 120) to yield a tensor of shape (28, 180, 150).
- Missing/inconsistent intraoperative values (<=0) are imputed with the patient's mean.
- Preoperative data (80 variables) are assumed to be present.
- Both intraoperative and preoperative features are standardized.
- The LS-LSTM model first extracts local features per segment via a bidirectional LSTM,
  then applies an inter-sequence LSTM to obtain a latent representation, which is fused with the preoperative vector.
- The training/dev split is 80/20.
"""

__author__ = 'hw56@iu.edu_improved'
__version__ = '0.3'
__license__ = 'MIT'

import argparse
import os
import pickle
from collections import Counter
from torch.optim.lr_scheduler import OneCycleLR

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
def impute_intraoperative(df, max_drop_rate=0.1, verbose=True):
    """
    For each intraoperative indicator, replace values <= 0 with the patient's mean for that indicator.
    Only drop an indicator if >max_drop_rate of its values are abnormal.

    Parameters:
    df (pandas.DataFrame): DataFrame containing patient data
    max_drop_rate (float): Maximum acceptable rate of abnormal values before dropping a column
    verbose (bool): Whether to print debug messages

    Returns:
    pandas.DataFrame: DataFrame with imputed values
    """
    dropped_cols = []
    for col in INTRA_FEATURES:
        if col not in df.columns:
            continue
        # Flag abnormal values (<= 0)
        abnormal_mask = df[col] <= 0
        abnormal_rate = abnormal_mask.mean()
        if abnormal_rate > max_drop_rate:
            # Drop the column entirely if too many abnormal values.
            df.drop(columns=[col], inplace=True)
            dropped_cols.append(col)
            if verbose:
                print(f"[DEBUG] Dropped {col} due to abnormal rate {abnormal_rate:.2f}")
        else:
            if abnormal_mask.any():
                # Impute with the patient's mean (computed ignoring abnormal values)
                valid_values = df.loc[~abnormal_mask, col]
                if len(valid_values) > 0:
                    valid_mean = valid_values.mean()
                    df.loc[abnormal_mask, col] = valid_mean
                else:
                    # If all values are abnormal, use global mean or 0
                    df.loc[abnormal_mask, col] = 0  # or some other reasonable default

    if verbose and dropped_cols:
        print(
            f"[WARNING] Dropped {len(dropped_cols)} columns due to high abnormal rates: {dropped_cols}")

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
                 preop_feature_means=None, preop_feature_stds=None, debug=False):
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
        self.debug = debug

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Enhanced __getitem__ method to handle missing features safely.
        """
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
        df = impute_intraoperative(df, verbose=False)

        # Select intraoperative columns that remain from INTRA_FEATURES
        intra_cols = [col for col in INTRA_FEATURES if col in df.columns]

        # Always create a fixed-size array with all expected features
        intra_array = np.zeros((df.shape[0], len(INTRA_FEATURES)), dtype=np.float32)

        # Fill in available features
        if len(intra_cols) > 0:
            for i, col in enumerate(INTRA_FEATURES):
                if col in df.columns:
                    intra_array[:, i] = df[col].values.astype(np.float32)

        # Handle NaN values
        intra_array = np.nan_to_num(intra_array, nan=0.0)

        # Further processing
        intra_array = pad_or_truncate_series(intra_array, self.intra_target_length)
        intra_tensor = segment_intraoperative(intra_array, seg_length=self.seg_length,
                                              hop_size=self.hop_size,
                                              pad_front=self.pad_front,
                                              pad_back=self.pad_back,
                                              num_segments=150)
        intra_tensor = torch.tensor(intra_tensor, dtype=torch.float)

        # Apply normalization if needed
        if self.normalize and self.intra_feature_means is not None and self.intra_feature_stds is not None:
            # Check if dimensions match before applying normalization
            if self.intra_feature_means.size(0) == intra_tensor.size(0):
                intra_tensor = (intra_tensor - self.intra_feature_means.view(-1, 1, 1)) / (
                        self.intra_feature_stds.view(-1, 1, 1) + 1e-8)
            else:
                print(f"[WARNING] Dimension mismatch: intra_tensor has {intra_tensor.size(0)} features, "
                      f"but means has {self.intra_feature_means.size(0)} features. Skipping normalization.")

        # Preoperative data
        preop_cols = [col for col in PREOP_FEATURES if col in df.columns]
        if len(preop_cols) == 0:
            preop_vector = torch.zeros(len(PREOP_FEATURES), dtype=torch.float)
        else:
            preop_df = df[preop_cols].copy()
            preop_array = np.zeros((preop_df.shape[0], len(PREOP_FEATURES)), dtype=np.float32)

            # Fill in available preoperative features
            for i, col in enumerate(PREOP_FEATURES):
                if col in preop_df.columns:
                    col_vals = preop_df[col].values.astype(np.float32)
                    if np.isnan(col_vals).any():
                        mean_val = np.nanmean(col_vals) if not np.isnan(np.nanmean(col_vals)) else 0.0
                        col_vals[np.isnan(col_vals)] = mean_val
                    preop_array[:, i] = col_vals

            preop_vector = torch.tensor(preop_array.mean(axis=0), dtype=torch.float)

        # Apply normalization for preoperative data if needed
        if self.normalize and self.preop_feature_means is not None and self.preop_feature_stds is not None:
            if self.preop_feature_means.size(0) == preop_vector.size(0):
                preop_vector = (preop_vector - self.preop_feature_means) / (
                        self.preop_feature_stds + 1e-8)

        # Print debug information for certain samples
        if self.debug and idx % 100 == 0:
            print(f"Sample {idx} intra_tensor stats: min={intra_tensor.min().item():.4f}, "
                  f"max={intra_tensor.max().item():.4f}, mean={intra_tensor.mean().item():.4f}")
            print(f"Sample {idx} preop stats: min={preop_vector.min().item():.4f}, "
                  f"max={preop_vector.max().item():.4f}, mean={preop_vector.mean().item():.4f}")

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
# Improved LS-LSTM Model with Attention Mechanism
##########################
class LS_LSTMClassifier(nn.Module):
    def __init__(self, intra_feat_dim=28, seg_length=180, num_segments=150, preop_dim=80,
                 hidden_size=128, num_classes=2, dropout=0.1, use_attention=True):
        """
        Improved LS-LSTM Model:
         - Intra-Sequence LSTM: Processes each segment (bidirectional) to extract a local feature vector.
         - Layer Normalization on the collected segment features.
         - Attention mechanism to focus on important segments
         - Inter-Sequence LSTM: Processes the sequence of segment features to output a latent intraoperative vector.
         - Fusion: Concatenate with preoperative vector.
         - Batch Normalization before final classification
         - FC: Final classification.
        """
        super(LS_LSTMClassifier, self).__init__()
        self.num_segments = num_segments
        self.hidden_size = hidden_size
        self.use_attention = use_attention

        # Improved initialization for better gradient flow
        self.intra_lstm = nn.LSTM(
            input_size=intra_feat_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Projection layer with skip connection
        self.intra_proj = nn.Linear(2 * hidden_size, hidden_size)

        # Layer normalization for better training stability
        self.layernorm = nn.LayerNorm(hidden_size)

        # Attention mechanism to focus on important segments
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1)
            )

        # Inter-sequence LSTM with residual connections
        self.inter_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # Batch normalization before final classification
        self.batch_norm = nn.BatchNorm1d(hidden_size + preop_dim)

        # Final classification layer
        self.fc = nn.Linear(hidden_size + preop_dim, num_classes)

        # Higher dropout for better regularization
        self.dropout = nn.Dropout(dropout)

        # SiLU activation (also known as Swish) for better gradient flow
        self.activation = nn.SiLU()

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better convergence"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM weights benefit from orthogonal initialization
                    nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:
                    # Linear layers use Kaiming initialization (only for tensors with at least 2 dims)
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
                else:
                    # For 1D parameters (like some batch norm weights)
                    nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, intra_tensor, preop):
        """
        intra_tensor: shape (batch, intra_feat_dim, seg_length, num_segments)
        preop: shape (batch, preop_dim)
        """
        B, F, L, S = intra_tensor.shape

        # Process each segment with intra-sequence LSTM
        x = intra_tensor.permute(0, 3, 2, 1).contiguous().view(B * S, L, F)
        lstm_out, (h_n, _) = self.intra_lstm(x)

        # Concatenate bidirectional hidden states
        h_n_cat = torch.cat([h_n[0], h_n[1]], dim=1)

        # Apply activation and projection with residual connection
        seg_features = self.activation(self.intra_proj(h_n_cat))

        # Reshape to (batch, segments, hidden)
        seg_features = seg_features.view(B, S, self.hidden_size)

        # Apply layer normalization
        seg_features = self.layernorm(seg_features)

        # Apply attention if enabled
        if self.use_attention:
            # Calculate attention scores and weights
            attention_scores = self.attention(seg_features)
            attention_weights = F.softmax(attention_scores, dim=1)

            # Apply attention weights
            context = torch.bmm(attention_weights.transpose(1, 2), seg_features)
            context = context.squeeze(1)

            # Skip connection
            inter_input = seg_features * attention_weights + seg_features
        else:
            inter_input = seg_features

        # Process with inter-sequence LSTM
        inter_out, (h_final, _) = self.inter_lstm(inter_input)
        intra_latent = h_final.squeeze(0)

        # Concatenate with preoperative data
        fused = torch.cat([intra_latent, preop], dim=1)

        # Apply batch normalization
        fused = self.batch_norm(fused)

        # Apply dropout and activation
        fused = self.dropout(self.activation(fused))

        # Final classification
        logits = self.fc(fused)

        return logits

    def encode(self, intra_tensor, preop):
        """Extract embeddings for triplet loss"""
        B, F, L, S = intra_tensor.shape
        x = intra_tensor.permute(0, 3, 2, 1).contiguous().view(B * S, L, F)
        lstm_out, (h_n, _) = self.intra_lstm(x)
        h_n_cat = torch.cat([h_n[0], h_n[1]], dim=1)
        seg_features = self.activation(self.intra_proj(h_n_cat))
        seg_features = seg_features.view(B, S, self.hidden_size)
        seg_features = self.layernorm(seg_features)

        if self.use_attention:
            attention_scores = self.attention(seg_features)
            attention_weights = F.softmax(attention_scores, dim=1)
            context = torch.bmm(attention_weights.transpose(1, 2), seg_features)
            context = context.squeeze(1)
            inter_input = seg_features * attention_weights + seg_features
        else:
            inter_input = seg_features

        inter_out, (h_final, _) = self.inter_lstm(inter_input)
        intra_latent = h_final.squeeze(0)
        fused = torch.cat([intra_latent, preop], dim=1)
        fused = self.batch_norm(fused)
        fused = self.dropout(self.activation(fused))
        return fused

##########################
# Training Loop with Early Stopping and Triplet Loss
##########################
from torch.nn import TripletMarginLoss

def train_model(model, train_loader, val_loader, device, epochs, learning_rate,
                class_weights, patience=5, grad_accum_steps=1, args=None):
    # Use AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=args.weight_decay if args else 0.0)

    # Use OneCycleLR scheduler for better convergence
    total_steps = len(train_loader) * epochs // grad_accum_steps
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=0.3,  # Spend 30% of training time warming up
        div_factor=10,  # Initial LR is max_lr/10
        final_div_factor=100  # Final LR is max_lr/1000
    )

    # Triplet loss with smaller margin for better separation
    triplet_criterion = torch.nn.TripletMarginLoss(
        margin=args.triplet_margin if args else 0.5)

    # Focal loss for handling class imbalance
    def focal_loss(logits, targets, class_weights=None, gamma=2.0):
        """Focal loss for handling class imbalance"""
        probs = F.softmax(logits, dim=1)
        ce_loss = F.cross_entropy(logits, targets, weight=class_weights, reduction='none')
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze()
        loss = ce_loss * ((1 - p_t) ** gamma)
        return loss.mean()

    # Track best validation metrics
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_val_auc = 0.0
    best_model_state = None
    no_improvement_count = 0

    # For gradient accumulation
    optimizer.zero_grad()

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_batch_count = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training", ncols=80, leave=False)):
            intra_tensor = batch["intra_tensor"].to(device)
            preop = batch["preop"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(intra_tensor, preop)
            logits = outputs

            # Use weighted focal loss if specified
            if args and args.focal_loss:
                ce_weight = class_weights.to(device) if args.weighted_ce else None
                loss = focal_loss(logits, labels, class_weights=ce_weight, gamma=args.focal_gamma)
            else:
                # Use vanilla or weighted cross-entropy
                ce_weight = class_weights.to(device) if (args and args.weighted_ce) else None
                ce_loss = F.cross_entropy(logits, labels, weight=ce_weight)
                loss = ce_loss

            # Add triplet loss if specified
            if args and args.triplet_loss:
                embeddings = model.encode(intra_tensor, preop)
                anchor_list, pos_list, neg_list = [], [], []

                # Sample triplets for triplet loss
                for i in range(len(labels)):
                    anchor_label = labels[i].item()
                    same_indices = (labels == anchor_label).nonzero(as_tuple=True)[0]
                    diff_indices = (labels != anchor_label).nonzero(as_tuple=True)[0]

                    # Skip if not enough samples for triplet
                    if len(same_indices) < 2 or len(diff_indices) < 1:
                        continue

                    # Sample positive example (same class but different sample)
                    pos_idx = np.random.choice(same_indices.cpu().numpy())
                    while pos_idx == i and len(same_indices) > 1:
                        pos_idx = np.random.choice(same_indices.cpu().numpy())

                    # Sample negative example (different class)
                    neg_idx = np.random.choice(diff_indices.cpu().numpy())

                    # Collect samples for triplet loss
                    anchor_list.append(embeddings[i].unsqueeze(0))
                    pos_list.append(embeddings[pos_idx].unsqueeze(0))
                    neg_list.append(embeddings[neg_idx].unsqueeze(0))

                # Calculate triplet loss if we have enough triplets
                if len(anchor_list) > 0:
                    anchor_t = torch.cat(anchor_list, dim=0)
                    pos_t = torch.cat(pos_list, dim=0)
                    neg_t = torch.cat(neg_list, dim=0)
                    tri_loss = triplet_criterion(anchor_t, pos_t, neg_t)

                    # Combine losses with weighting
                    loss = ce_loss + args.triplet_weight * tri_loss

                    # Log triplet loss
                    wandb.log({"triplet_loss": tri_loss.item()})

            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps

            # Backward pass
            loss.backward()

            # Track losses
            train_losses.append(loss.item() * grad_accum_steps)
            train_batch_count += 1

            # Check gradient norms before clipping (for monitoring)
            if batch_idx % grad_accum_steps == 0 or batch_idx % 20 == 0:
                grad_norms = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        norm = param.grad.data.norm(2).item()
                        grad_norms.append(norm)

                avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
                max_grad_norm = np.max(grad_norms) if grad_norms else 0.0

                print(f"Epoch {epoch}: Average Grad Norm = {avg_grad_norm:.4f}, Max Grad Norm = {max_grad_norm:.4f}")
                wandb.log({
                    "avg_grad_norm": avg_grad_norm,
                    "max_grad_norm": max_grad_norm
                })

            # Gradient accumulation step
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Log learning rate
                wandb.log({"learning_rate": scheduler.get_last_lr()[0]})

        # Calculate average train loss
        avg_train_loss = np.mean(train_losses)

        # Validation phase
        model.eval()
        val_losses = []
        all_probs = []  # For AUC calculation
        all_preds = []  # For binary classification metrics
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", ncols=80, leave=False):
                intra_tensor = batch["intra_tensor"].to(device)
                preop = batch["preop"].to(device)
                labels = batch["label"].to(device)

                # Forward pass
                outputs = model(intra_tensor, preop)

                # Compute loss for validation
                ce_weight = class_weights.to(device) if (args and args.weighted_ce) else None
                val_loss = F.cross_entropy(outputs, labels, weight=ce_weight)
                val_losses.append(val_loss.item())

                # Get probability for the positive class
                probs = F.softmax(outputs, dim=1)[:, 1]

                # Get binary predictions
                preds = torch.argmax(outputs, dim=1)

                # Collect results for metrics calculation
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate validation metrics
        avg_val_loss = np.mean(val_losses)
        acc = accuracy_score(all_labels, all_preds)

        # Calculate AUC (safely)
        # try:
        auc = roc_auc_score(all_labels, all_probs)
        # except Exception as e:
        #     print(f"[WARNING] AUC calculation error: {e}. Using default value of 0.5.")
        #     auc = 0.5

        # Calculate precision, recall, f1 using binary predictions
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary', zero_division=0
        )

        # Print epoch results
        print(
            f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, "
            f"Acc = {acc:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}, "
            f"Precision = {precision:.4f}, Recall = {recall:.4f}"
        )

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "accuracy": acc,
            "auc": auc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "class_distribution": wandb.Histogram(np.array(all_labels))
        })

        # Early stopping logic based on multiple metrics
        improved = False

        # Check for improvement in validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            improved = True
            print(f"New best validation loss: {best_val_loss:.4f}")

        # Check for improvement in F1 score (useful for imbalanced datasets)
        if f1 > best_val_f1:
            best_val_f1 = f1
            improved = True
            print(f"New best validation F1: {best_val_f1:.4f}")

        # Check for improvement in AUC (robust to imbalanced classes)
        if auc > best_val_auc:
            best_val_auc = auc
            improved = True
            print(f"New best validation AUC: {best_val_auc:.4f}")

        # Save best model if any metric improved
        if improved:
            best_model_state = model.state_dict()

            # Save checkpoint
            if args and args.save_checkpoints:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_f1': f1,
                    'val_auc': auc,
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

            no_improvement_count = 0
        else:
            no_improvement_count += 1
            print(f"No improvement for {no_improvement_count} epochs")

            if no_improvement_count >= patience:
                print(f"Early stopping triggered at epoch {epoch} after {patience} epochs with no improvement.")
                break

    # Load best model state at the end
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model state from memory")

    return model



##########################
# Functions to compute normalization statistics for both intra and preop data.
##########################
def compute_normalization_stats(dataset, key, feature_dim=None):
    """
    Compute normalization statistics (mean and standard deviation) for the given dataset and key.

    Parameters:
    dataset: The dataset to compute statistics for
    key: Either "intra_tensor" or "preop"
    feature_dim: Expected feature dimension (default: None)

    Returns:
    means, stds: Tensors containing means and standard deviations
    """
    print(f"Computing normalization stats for {key}")

    # Use a small subset for initial computation to check dimensions
    try:
        sample_idx = min(10, len(dataset) - 1)
        sample = dataset[sample_idx][key]
        print(f"Sample {key} shape: {sample.shape}")
    except Exception as e:
        print(f"[WARNING] Error checking sample shape: {e}")

    all_data = []
    for i in tqdm(range(len(dataset)), desc=f"Computing normalization stats for {key}"):
        try:
            sample = dataset[i][key]
            all_data.append(sample)
        except Exception as e:
            print(f"[WARNING] Error in sample {i}: {e}")
            continue

    if len(all_data) == 0:
        print(f"[ERROR] No valid samples found for {key}")
        # Return default values
        if key == "intra_tensor":
            return torch.zeros(28), torch.ones(28)
        else:
            return torch.zeros(80), torch.ones(80)

    try:
        all_data = torch.stack(all_data, dim=0)
        print(f"Stacked {key} data shape: {all_data.shape}")

        if key == "intra_tensor":
            # For intra_tensor: (batch, features, seg_length, num_segments)
            means = all_data.mean(dim=(0, 2, 3))
            stds = all_data.std(dim=(0, 2, 3))
            stds[stds < 1e-6] = 1.0  # Replace near-zero stds with 1 to avoid division by zero
        else:
            # For preop: (batch, features)
            means = all_data.mean(dim=0)
            stds = all_data.std(dim=0)
            stds[stds < 1e-6] = 1.0

        print(f"Computed {key} stats - means shape: {means.shape}, stds shape: {stds.shape}")

        # Ensure correct dimensions if specified
        if feature_dim is not None:
            if means.size(0) != feature_dim:
                print(f"[WARNING] Expected {feature_dim} features for {key}, but got {means.size(0)}. Padding.")
                if means.size(0) < feature_dim:
                    # Pad with zeros/ones if fewer features than expected
                    means = torch.cat([means, torch.zeros(feature_dim - means.size(0))])
                    stds = torch.cat([stds, torch.ones(feature_dim - stds.size(0))])
                else:
                    # Truncate if more features than expected
                    means = means[:feature_dim]
                    stds = stds[:feature_dim]

        return means, stds

    except Exception as e:
        print(f"[ERROR] Failed to compute statistics: {e}")
        # Return default values
        if key == "intra_tensor":
            return torch.zeros(28), torch.ones(28)
        else:
            return torch.zeros(80), torch.ones(80)


def load_or_compute_norm_stats(args, train_dataset):
    """
    Load normalization statistics from file or compute them if not available.

    Parameters:
    args: Command line arguments
    train_dataset: Training dataset

    Returns:
    intra_means, intra_stds, preop_means, preop_stds: Normalization statistics
    """
    norm_stats_file = os.path.join(args.output_dir, "normalization_stats.pkl")

    # Option to force recomputation
    force_recompute = not os.path.exists(norm_stats_file) or args.recompute_stats

    if not force_recompute:
        try:
            with open(norm_stats_file, "rb") as f:
                stats = pickle.load(f)

            intra_means = stats["intra_means"]
            intra_stds = stats["intra_stds"]
            preop_means = stats["preop_means"]
            preop_stds = stats["preop_stds"]

            # Verify dimensions
            if (intra_means.size(0) != len(INTRA_FEATURES) or
                    preop_means.size(0) != len(PREOP_FEATURES)):
                print("[WARNING] Loaded stats have incorrect dimensions. Recomputing...")
                force_recompute = True
            else:
                print(f"[INFO] Loaded normalization stats from {norm_stats_file}")
                print(f"[INFO] Intra means shape: {intra_means.shape}, Preop means shape: {preop_means.shape}")
                return intra_means, intra_stds, preop_means, preop_stds

        except Exception as e:
            print(f"[WARNING] Failed to load normalization stats: {e}. Recomputing...")
            force_recompute = True

    # Compute statistics
    print("[INFO] Computing normalization statistics...")
    intra_means, intra_stds = compute_normalization_stats(
        train_dataset, key="intra_tensor", feature_dim=len(INTRA_FEATURES))
    preop_means, preop_stds = compute_normalization_stats(
        train_dataset, key="preop", feature_dim=len(PREOP_FEATURES))

    # Save statistics
    stats = {
        "intra_means": intra_means,
        "intra_stds": intra_stds,
        "preop_means": preop_means,
        "preop_stds": preop_stds
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(norm_stats_file, "wb") as f:
        pickle.dump(stats, f)

    print(f"[INFO] Saved normalization stats to {norm_stats_file}")
    print(f"[INFO] Intra means shape: {intra_means.shape}, Preop means shape: {preop_means.shape}")

    return intra_means, intra_stds, preop_means, preop_stds

##########################
# Improved class weights calculation function
##########################
def compute_class_weights(labels):
    """
    Compute balanced class weights with smoothing to avoid extreme values.

    Parameters:
    labels: List or array of class labels

    Returns:
    torch.Tensor: Class weights tensor
    """
    unique, counts = np.unique(labels, return_counts=True)
    n_samples = len(labels)
    n_classes = len(unique)

    if n_classes < 2:
        return torch.tensor([1.0, 1.0], dtype=torch.float)

    # Calculate inverse frequency
    weights = n_samples / (counts * n_classes)

    # Apply square root to smooth extreme weights
    weights = np.sqrt(weights)

    # Normalize weights for balanced contribution
    weights = weights / weights.sum() * n_classes

    # Create a balanced weight dictionary
    weight_dict = {int(k): v for k, v in zip(unique, weights)}

    # Create tensor with weights for all classes
    return torch.tensor([weight_dict.get(i, 1.0) for i in range(max(2, n_classes))], dtype=torch.float)

##########################
# Main Function (Improved version)
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
    if args.use_attention:
        run_name += "_ATT"
    if args.focal_loss:
        run_name += f"_FL{args.focal_gamma}"

    # Initialize wandb with more configuration options
    wandb.init(
        project=f"{args.disease.upper()}_LSLSTM",
        name=run_name,
        config=vars(args),
        tags=[args.disease, "improved", "ls-lstm"]
    )

    # Set device with better error handling
    if args.no_cuda:
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            print("[WARNING] CUDA requested but not available. Using CPU instead.")
            device = torch.device("cpu")
        else:
            try:
                device = torch.device(f"cuda:{args.cuda}")
                torch.cuda.set_device(args.cuda)
            except Exception as e:
                print(f"[ERROR] Failed to set CUDA device {args.cuda}: {e}. Using CPU instead.")
                device = torch.device("cpu")

    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load label file with better error handling
    try:
        df_labels = pd.read_excel(args.label_file)
        df_labels = df_labels.drop_duplicates().dropna(subset=[disease_col, "ID"])
        label_dict = {str(x).strip(): int(y) for x, y in zip(df_labels["ID"], df_labels[disease_col])}

        # Print label distribution
        label_counts = Counter([int(v) for v in label_dict.values()])
        print("Label distribution in full dataset:", label_counts)

        # Check for extreme class imbalance
        min_class_count = min(label_counts.values())
        if min_class_count < 10:
            print(f"[WARNING] Very small minority class detected ({min_class_count} samples).")
            print("Consider data augmentation or specialized handling.")

    except Exception as e:
        raise RuntimeError(f"Error loading label file: {e}")

    # Find patient files with more robust file handling
    try:
        if not os.path.exists(args.data_dir):
            raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

        file_list = []
        for fname in os.listdir(args.data_dir):
            if fname.endswith(".csv"):
                patient_id = fname.split("_")[0].strip()
                if patient_id in label_dict:
                    file_path = os.path.join(args.data_dir, fname)
                    # Quick validation check
                    try:
                        if os.path.getsize(file_path) > 0:
                            file_list.append(file_path)
                        else:
                            print(f"[WARNING] Empty file: {fname}")
                    except Exception:
                        print(f"[WARNING] Cannot access file: {fname}")

        if args.debug:
            file_list = file_list[: args.max_patients]

        if len(file_list) == 0:
            raise ValueError("No valid patient files found.")

        print(f"Found {len(file_list)} patient files.")

    except Exception as e:
        raise RuntimeError(f"Error finding patient files: {e}")

    # For LS-LSTM, we fix intraoperative sequence length to 18000 seconds.
    intra_target_length = 18000

    # Preprocess files and save concatenated DataFrame (optional)
    processed_samples = []

    print("Preprocessing patient files...")
    for f in tqdm(file_list, desc="Preprocessing", ncols=80):
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

        # Use the improved impute_intraoperative function
        df = impute_intraoperative(df, max_drop_rate=args.max_drop_rate, verbose=args.debug)
        processed_samples.append(df)

    if len(processed_samples) == 0:
        raise ValueError("No patient files processed successfully.")

    all_patients_df = pd.concat(processed_samples, ignore_index=True)
    preprocessed_path = os.path.join(args.output_dir, f"{args.disease}_preprocessed.parquet")
    pq.write_table(pa.Table.from_pandas(all_patients_df), preprocessed_path)
    print(f"Preprocessed data saved to {preprocessed_path}.")

    # Improved stratified split with validation
    unique_ids = all_patients_df["ID"].unique()
    labels_for_split = [label_dict[str(pid).strip()] for pid in unique_ids]

    # Check for extreme imbalance
    class_counts = Counter(labels_for_split)
    print(f"Class distribution in split: {class_counts}")

    # Stratified split
    train_ids, val_ids = train_test_split(
        unique_ids,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=labels_for_split
    )

    print(f"Training patients: {len(train_ids)}, Validation patients: {len(val_ids)}")

    # Create training dataset with debug option
    train_dataset = PatientTimeSeriesDataset(
        file_list=[f for f in file_list if os.path.basename(f).split("_")[0].strip() in train_ids],
        label_dict=label_dict,
        disease_col=disease_col,
        intra_target_length=intra_target_length,
        process_mode="lslstm",
        normalize=False,
        debug=args.debug
    )

    val_dataset = PatientTimeSeriesDataset(
        file_list=[f for f in file_list if os.path.basename(f).split("_")[0].strip() in val_ids],
        label_dict=label_dict,
        disease_col=disease_col,
        intra_target_length=intra_target_length,
        process_mode="lslstm",
        normalize=False,
        debug=args.debug
    )

    # Use the improved normalization stats loader with robust error handling
    try:
        intra_means, intra_stds, preop_means, preop_stds = load_or_compute_norm_stats(args, train_dataset)
    except Exception as e:
        print(f"[ERROR] Failed to compute normalization stats: {e}")
        print("Using default normalization (zero mean, unit std)")
        intra_means = torch.zeros(len(INTRA_FEATURES))
        intra_stds = torch.ones(len(INTRA_FEATURES))
        preop_means = torch.zeros(len(PREOP_FEATURES))
        preop_stds = torch.ones(len(PREOP_FEATURES))

    # Apply normalization to datasets
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

    # Now safely get all training labels
    all_train_labels = []
    for i in tqdm(range(len(train_dataset)), desc="Collecting training labels", ncols=80):
        try:
            label = train_dataset[i]["label"].item()
            all_train_labels.append(label)
        except Exception as e:
            print(f"[WARNING] Error getting label for sample {i}: {e}")

    print("Training label distribution (dataset):", dict(Counter(all_train_labels)))

    # Configure data loading with oversampling if needed
    if args.oversample:
        print("[INFO] Using oversampling (WeightedRandomSampler).")
        labels_np = np.array(all_train_labels)
        unique_labels, counts = np.unique(labels_np, return_counts=True)

        # Calculate balanced weights
        weight_per_class = 1.0 / len(unique_labels)
        samples_per_class = weight_per_class / (counts / sum(counts))

        # Assign weights to each sample
        weights = np.zeros_like(labels_np, dtype=np.float32)
        for idx, label in enumerate(unique_labels):
            weights[labels_np == label] = samples_per_class[idx]

        train_sampler = WeightedRandomSampler(
            weights,
            num_samples=int(len(labels_np) * args.oversample_factor),
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=custom_data_collator,
            pin_memory=True if device.type == 'cuda' else False,
            drop_last=args.drop_last
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=custom_data_collator,
            pin_memory=True if device.type == 'cuda' else False,
            drop_last=args.drop_last
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_data_collator,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Compute improved class weights
    class_weights = compute_class_weights(all_train_labels)
    print("Computed class weights (balanced):", class_weights)

    num_intra_channels = len(INTRA_FEATURES)
    num_preop = len(PREOP_FEATURES)
    print(f"Number of intraoperative features: {num_intra_channels}")
    print(f"Number of preoperative features: {num_preop}")

    # Create model with improved architecture
    model = LS_LSTMClassifier(
        intra_feat_dim=num_intra_channels,
        seg_length=180,
        num_segments=150,
        preop_dim=num_preop,
        hidden_size=args.hidden_size,
        num_classes=2,
        dropout=args.dropout,
        use_attention=args.use_attention
    ).to(device)

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters, {trainable_params:,} trainable")

    # Create checkpoint directory
    ckpt_folder = os.path.join(args.output_dir, run_name)
    os.makedirs(ckpt_folder, exist_ok=True)

    # Train the model with improved training loop
    model = train_model(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.learning_rate,
        class_weights,
        patience=args.patience,
        grad_accum_steps=args.grad_accum_steps,
        args=args
    )

    # Final evaluation on validation set
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    print("Performing final evaluation...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final Evaluation", ncols=80):
            intra_tensor = batch["intra_tensor"].to(device)
            preop = batch["preop"].to(device)
            labels = batch["label"].to(device)
            outputs = model(intra_tensor, preop)

            # Get probability for the positive class
            probs = F.softmax(outputs, dim=1)[:, 1]

            # Get binary predictions
            preds = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate final metrics
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )

    print(f"Final Evaluation: Accuracy = {acc:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}")
    print(f"Precision = {precision:.4f}, Recall = {recall:.4f}")

    # Save final model
    final_model_path = os.path.join(ckpt_folder, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_acc': acc,
        'val_auc': auc,
        'val_f1': f1,
        'val_precision': precision,
        'val_recall': recall,
        'intra_feature_means': intra_means,
        'intra_feature_stds': intra_stds,
        'preop_feature_means': preop_means,
        'preop_feature_stds': preop_stds
    }, final_model_path)

    print(f"Model saved to {final_model_path}")

    # Log final metrics to wandb
    wandb.log({
        "final_accuracy": acc,
        "final_auc": auc,
        "final_f1": f1,
        "final_precision": precision,
        "final_recall": recall,
        "checkpoint_path": final_model_path
    })

    # Finish wandb logging
    wandb.finish()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved LS-LSTM for Disease Prediction")

    # Data parameters
    parser.add_argument("--data_dir", type=str, default="time_series_data_LSTM_10_29_2024",
                        help="Directory containing patient CSV files.")
    parser.add_argument("--label_file", type=str, default="imputed_demo_data.xlsx",
                        help="Excel file containing patient labels.")
    parser.add_argument("--process_mode", type=str, choices=["lslstm", "pool", "truncate", "none"],
                        default="lslstm", help="Preprocessing mode.")
    parser.add_argument("--max_drop_rate", type=float, default=0.1,
                        help="Maximum rate of abnormal values before dropping a feature column.")

    # Dataset splitting
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of data to use for validation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Mini-batch size for training.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                        help="Learning rate for optimizer.")
    parser.add_argument("--patience", type=int, default=20,
                        help="Patience for early stopping.")
    parser.add_argument("--grad_accum_steps", type=int, default=2,
                        help="Number of gradient accumulation steps.")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0,
                        help="Max norm for gradient clipping.")

    # Model architecture
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden size for LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate.")
    parser.add_argument("--use_attention", action="store_true",
                        help="Use attention mechanism in the model.")

    # Loss function parameters
    parser.add_argument("--weighted_ce", action="store_true",
                        help="Use class-weighted cross-entropy loss.")
    parser.add_argument("--focal_loss", action="store_true",
                        help="Use focal loss instead of cross-entropy.")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma parameter for focal loss.")
    parser.add_argument("--triplet_loss", action="store_true",
                        help="Use triplet loss in addition to classification loss.")
    parser.add_argument("--triplet_margin", type=float, default=0.5,
                        help="Margin for triplet loss.")
    parser.add_argument("--triplet_weight", type=float, default=0.5,
                        help="Weight for triplet loss relative to classification loss.")

    # Regularization
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay (L2 regularization).")

    # Data sampling
    parser.add_argument("--oversample", action="store_true",
                        help="Use weighted random sampling to balance classes.")
    parser.add_argument("--oversample_factor", type=float, default=1.0,
                        help="Factor to multiply dataset size when oversampling.")
    parser.add_argument("--drop_last", action="store_true",
                        help="Drop last incomplete batch during training.")

    # Hardware & debug options
    parser.add_argument("--cuda", type=int, default=0,
                        help="GPU device index to use.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for data loading.")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: process fewer patients and print extra info.")
    parser.add_argument("--max_patients", type=int, default=100,
                        help="Max patients to process in debug mode.")

    # Output options
    parser.add_argument("--output_dir", type=str, default="./lslstm_improved",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--save_checkpoints", action="store_true",
                        help="Save checkpoints during training.")
    parser.add_argument("--recompute_stats", action="store_true",
                        help="Force recomputation of normalization statistics.")

    # Disease argument
    parser.add_argument("--disease", type=str, choices=["aki", "af", "pneumonia", "pd", "pod30"],
                        default="aki", help="Which disease label to predict.")

    args = parser.parse_args()
    main(args)