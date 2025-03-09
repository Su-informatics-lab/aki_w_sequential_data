#!/usr/bin/env python
"""
PatchTST Classification for AKI Prediction via LSTM

This script builds a time series classification model using a custom LSTM architecture to predict
Acute Kidney Injury (AKI) from multi-channel patient vital sign data. The raw data are stored as individual CSV files,
each corresponding to one patient. The patient ID is inferred from the filename and is used to look up the static AKI label
from an Excel file.

Overall Process:
+--------------------------------------------------------------+
|                      Raw CSV Files                           |
|  Each patient CSV contains a time series (e.g., one sample per second) with 26 channels. |
|  Patient ID is inferred from the filename (e.g., "R94657_combined.csv" → "R94657").         |
+--------------------------------------------------------------+
              |
              | Preprocessing (minute-pooling / truncation) per CSV.
              | Add "ID" and "time_idx" columns.
              | Look up AKI label from Excel via patient ID.
              v
+--------------------------------------------------------------+
|       Combined Preprocessed Data (Parquet)                   |
| Columns: ID, Acute_kidney_injury, time_idx, F1, F2, …, F26     |
+--------------------------------------------------------------+
              |
              | Filter for valid IDs and split by patient ID into train and validation sets.
              v
+--------------------------------------------------------------+
|      ForecastDFDataset (from tsfm_public.toolkit.dataset)      |
|  Configured with:                                              |
|     id_columns = ["ID"], timestamp_column = "time_idx",        |
|     target_columns = observable_columns = [F1, F2, …, F26]      |
|  (Only the observable signals are used as input to the model)    |
+--------------------------------------------------------------+
              |
              | Wrap with ClassificationDataset to add a static "labels" key (AKI label) based on patient ID.
              v
+--------------------------------------------------------------+
|         ClassificationDataset (Custom Wrapper)               |
|  Each example is a dict containing:                          |
|    - past_values: tensor of shape [context_length, 26]         |
|    - labels: scalar (AKI label from Excel)                     |
+--------------------------------------------------------------+
              |
              | DataLoader batches examples using a custom collator.
              v
+--------------------------------------------------------------+
|         AKI_LSTMClassifier Model (LSTM-based)                |
|  Input: [batch, context_length, 26] observable channels         |
|  Uses a 2-layer LSTM with dropout, SiLU activation, and a       |
|  residual connection to output logits for binary classification.|
+--------------------------------------------------------------+
              |
              | Loss computed using the provided "labels"
              v
+--------------------------------------------------------------+
|        Training via CustomTrainer (with EarlyStopping)         |
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import wandb
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from collections import Counter

# Import ForecastDFDataset from tsfm_public toolkit.
from tsfm_public.toolkit.dataset import ForecastDFDataset

from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from transformers.configuration_utils import PretrainedConfig

# Patch PretrainedConfig's to_dict for JSON serialization.
original_to_dict = PretrainedConfig.to_dict
def patched_to_dict(self):
    output = original_to_dict(self)
    for key, value in output.items():
        if isinstance(value, np.integer):
            output[key] = int(value)
        elif isinstance(value, np.floating):
            output[key] = float(value)
        elif isinstance(value, np.ndarray):
            output[key] = value.tolist()
    return output
PretrainedConfig.to_dict = patched_to_dict

###########################
#  LSTM Model Definition  #
###########################
class AKI_LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        """
        LSTM-based classifier for AKI prediction.
        Args:
            input_size (int): Number of observable channels.
            hidden_size (int): Hidden dimension of the LSTM.
            num_layers (int): Number of LSTM layers.
            num_classes (int): Number of output classes (2 for binary classification).
            dropout (float): Dropout probability.
        """
        super(AKI_LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # Residual connection: if input_size != hidden_size, project input to hidden_size.
        self.residual = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [B, T, C]
        out, (h_n, c_n) = self.lstm(x)  # h_n: [num_layers, B, hidden_size]
        h_last = h_n[-1]  # [B, hidden_size]
        if self.residual is not None:
            # Use mean pooling over time for residual connection
            residual = self.residual(x.mean(dim=1))
            h_last = h_last + residual
        h_last = self.activation(h_last)
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)
        return logits

#############################
#   Data Preprocessing      #
#############################
def pool_minute(df, pool_window=60, pool_method="average"):
    """Pool over non-overlapping windows using specified method."""
    exclude_cols = {"ID", "Acute_kidney_injury", "time_idx"}
    feature_cols = [col for col in df.columns if col not in exclude_cols and np.issubdtype(df[col].dtype, np.number)]
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
    """Truncate or pad df to exactly fixed_length rows."""
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

class OnTheFlyForecastDFDataset(Dataset):
    """
    Dataset for on-the-fly processing of patient CSV files.
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
        # Infer patient ID from file name (e.g., "R94657_combined.csv" -> "R94657")
        patient_id = fname.split('_')[0].strip()
        df = pd.read_csv(csv_path)
        df["time_idx"] = range(len(df))
        df["ID"] = patient_id
        # Use label mapping; if missing, np.nan is returned.
        df["Acute_kidney_injury"] = int(self.label_dict.get(patient_id, np.nan))
        if self.process_mode == "truncate":
            df = truncate_pad_series(df, fixed_length=self.fixed_length)
        elif self.process_mode == "pool":
            df = pool_minute(df, pool_window=self.pool_window, pool_method=self.pool_method)
        df["Acute_kidney_injury"] = df["Acute_kidney_injury"].astype(np.int64)
        return df

#############################
#   Classification Dataset  #
#############################
class ClassificationDataset(Dataset):
    """
    A wrapper around ForecastDFDataset that adds a static "labels" key to each example
    based on the patient ID.
    """
    def __init__(self, base_dataset, label_mapping, debug=False):
        self.base_dataset = base_dataset
        self.label_mapping = {str(k).strip(): v for k, v in label_mapping.items()}
        self.debug = debug

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        example = dict(self.base_dataset[idx])
        # Use "ID" as patient identifier
        patient_id = example.get("ID")
        if patient_id is None:
            raise KeyError("No patient ID found in example.")
        patient_id = str(patient_id).strip()
        if patient_id not in self.label_mapping:
            raise KeyError(f"Patient ID {patient_id} not found in label mapping.")
        example["labels"] = torch.tensor(self.label_mapping[patient_id], dtype=torch.long)
        if self.debug and idx == 0:
            print(f"[DEBUG] ClassificationDataset example keys: {list(example.keys())}")
            print(f"[DEBUG] Patient ID: {patient_id}, Label: {example['labels']}")
        return example

#############################
#   Custom Data Collator    #
#############################
def custom_data_collator(features):
    """
    Custom collate function to ensure that the "labels" key is preserved.
    """
    all_keys = set()
    for f in features:
        all_keys.update(f.keys())
    batch = {}
    for key in all_keys:
        if key == "labels":
            batch[key] = torch.tensor([f.get(key) for f in features], dtype=torch.long)
        else:
            try:
                batch[key] = torch.stack([f.get(key) for f in features])
            except Exception as e:
                print(f"[DEBUG] Could not stack key '{key}': {e}")
                batch[key] = [f.get(key) for f in features]
    print(f"[DEBUG] Collated batch keys: {list(batch.keys())}")
    return batch

#############################
#   Custom Trainer          #
#############################
class LSTMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        print("[DEBUG] Batch keys before loss computation:", list(inputs.keys()))
        labels = inputs.pop("labels").long()
        # For LSTM classifier, input is just "past_values"
        outputs = model(inputs["past_values"])
        if self.class_weights is not None:
            loss = F.cross_entropy(outputs, labels, weight=self.class_weights.to(labels.device))
        else:
            loss = F.cross_entropy(outputs, labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            labels = inputs.pop("labels").long()
            outputs = model(inputs["past_values"])
            loss = F.cross_entropy(outputs, labels, weight=self.class_weights.to(labels.device) if self.class_weights is not None else None)
            return (loss, outputs, labels)

#############################
#      LSTM Model           #
#############################
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
        # Residual connection if input size differs from hidden_size
        self.residual = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [B, T, C]
        out, (h_n, c_n) = self.lstm(x)
        h_last = h_n[-1]  # [B, hidden_size]
        if self.residual is not None:
            residual = self.residual(x.mean(dim=1))
            h_last = h_last + residual
        h_last = self.activation(h_last)
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)
        return logits

#############################
#         Main              #
#############################
def main(args):
    wandb.init(project="patchtst_aki", config=vars(args))
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Load AKI labels from Excel
    df_labels = pd.read_excel("imputed_demo_data.xlsx")
    df_labels = df_labels[["ID", "Acute_kidney_injury"]].drop_duplicates().dropna(subset=["Acute_kidney_injury"])
    label_dict = {str(x).strip(): int(y) for x, y in zip(df_labels["ID"], df_labels["Acute_kidney_injury"])}

    # Filter file list
    file_list = [
        os.path.join(args.data_dir, fname)
        for fname in os.listdir(args.data_dir)
        if fname.endswith(".csv") and fname.split('_')[0].strip() in label_dict
    ]
    if args.debug:
        file_list = file_list[:args.max_patients]

    preprocessed_path = args.preprocessed_path
    if os.path.exists(preprocessed_path):
        print(f"Loading preprocessed data from {preprocessed_path}...")
        all_patients_df = pd.read_parquet(preprocessed_path)
    else:
        base_dataset = OnTheFlyForecastDFDataset(
            file_list=file_list,
            label_dict=label_dict,
            process_mode=args.process_mode,
            pool_window=args.pool_window,
            pool_method=args.pool_method,
            fixed_length=args.fixed_length
        )
        print("Processing patient files...")
        writer = None
        for i in tqdm(range(len(base_dataset))):
            df = base_dataset[i]
            df["Acute_kidney_injury"] = df["Acute_kidney_injury"].astype(np.int64)
            table = pa.Table.from_pandas(df)
            if writer is None:
                writer = pq.ParquetWriter(preprocessed_path, table.schema)
            else:
                table = table.cast(writer.schema)
            writer.write_table(table)
        if writer is not None:
            writer.close()
        all_patients_df = pd.read_parquet(preprocessed_path)
        print(f"Preprocessed data saved to {preprocessed_path}.")

    print("Columns in preprocessed data:", all_patients_df.columns.tolist())
    all_patients_df["id"] = all_patients_df["ID"]
    all_patients_df = all_patients_df[all_patients_df["ID"].isin(label_dict.keys())]

    # Split data by patient ID using stratification
    unique_ids = all_patients_df["ID"].unique()
    labels = [label_dict[str(pid).strip()] for pid in unique_ids]
    train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=42, stratify=labels)
    print("Training set label distribution:", Counter([label_dict[str(pid).strip()] for pid in train_ids]))
    print("Validation set label distribution:", Counter([label_dict[str(pid).strip()] for pid in val_ids]))

    train_df = all_patients_df[all_patients_df["ID"].isin(train_ids)]
    val_df = all_patients_df[all_patients_df["ID"].isin(val_ids)]

    print("Training label distribution (from data):", train_df["Acute_kidney_injury"].value_counts().to_dict())
    print("Validation label distribution (from data):", val_df["Acute_kidney_injury"].value_counts().to_dict())

    # Determine observable features: exclude ID, target, time_idx, and 'id'
    feature_cols = [col for col in all_patients_df.columns
                    if col not in {"ID", "Acute_kidney_injury", "time_idx", "id"}
                    and not col.startswith("Unnamed")
                    and np.issubdtype(all_patients_df[col].dtype, np.number)]
    print(f"Detected {len(feature_cols)} feature channels: {feature_cols}")
    print("Using only observable features as model input (do not include the target).")
    num_input_channels = len(feature_cols)
    print(f"Number of input channels (observable features): {num_input_channels}")

    history_length = train_df.groupby("ID").size().max()
    print(f"History length (number of rows per patient): {history_length}")

    # Create ForecastDFDataset objects that use only observable features.
    base_train_dataset = ForecastDFDataset(
        data=train_df,
        id_columns=["ID"],
        timestamp_column="time_idx",
        target_columns=feature_cols,
        observable_columns=feature_cols,
        context_length=history_length,
        prediction_length=1,
        static_categorical_columns=["Acute_kidney_injury"],
    )
    base_val_dataset = ForecastDFDataset(
        data=val_df,
        id_columns=["ID"],
        timestamp_column="time_idx",
        target_columns=feature_cols,
        observable_columns=feature_cols,
        context_length=history_length,
        prediction_length=1,
        static_categorical_columns=["Acute_kidney_injury"],
    )
    # Wrap with ClassificationDataset to add static AKI label
    from collections import defaultdict

    class ClassificationDataset(Dataset):
        """
        Wraps ForecastDFDataset to add a static 'labels' key for classification.
        """
        def __init__(self, base_dataset, label_mapping, debug=False):
            self.base_dataset = base_dataset
            self.label_mapping = {str(k).strip(): v for k, v in label_mapping.items()}
            self.debug = debug

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            example = dict(self.base_dataset[idx])
            patient_id = example.get("ID", None)
            if patient_id is None:
                raise KeyError("No patient ID found in example.")
            patient_id = str(patient_id).strip()
            if patient_id not in self.label_mapping:
                raise KeyError(f"Patient ID {patient_id} not found in label mapping.")
            example["labels"] = torch.tensor(self.label_mapping[patient_id], dtype=torch.long)
            if self.debug and idx == 0:
                print(f"[DEBUG] ClassificationDataset example keys: {list(example.keys())}")
                print(f"[DEBUG] Patient ID: {patient_id}, Label: {example['labels']}")
            return example

    train_dataset = ClassificationDataset(base_train_dataset, label_dict, debug=args.debug)
    val_dataset = ClassificationDataset(base_val_dataset, label_dict, debug=args.debug)

    # Create DataLoaders using custom collate function.
    def custom_data_collator(features):
        all_keys = set()
        for f in features:
            all_keys.update(f.keys())
        batch = {}
        for key in all_keys:
            if key == "labels":
                batch[key] = torch.tensor([f.get(key) for f in features], dtype=torch.long)
            else:
                try:
                    batch[key] = torch.stack([f.get(key) for f in features])
                except Exception as e:
                    print(f"[DEBUG] Could not stack key '{key}': {e}")
                    batch[key] = [f.get(key) for f in features]
        print(f"[DEBUG] Collated batch keys: {list(batch.keys())}")
        return batch

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=custom_data_collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=custom_data_collator)

    # Compute class weights
    labels_train = train_df["Acute_kidney_injury"].values
    print("Training label distribution:", dict(Counter(labels_train)))
    num_classes = 2
    def compute_class_weights(labels):
        unique, counts = np.unique(labels, return_counts=True)
        n_samples = len(labels)
        n_classes = len(unique)
        if n_classes < 2:
            return torch.tensor([1.0, 1.0], dtype=torch.float)
        weights = n_samples / (n_classes * counts)
        weight_dict = {class_val: weight for class_val, weight in zip(unique, weights)}
        return torch.tensor([weight_dict.get(i, 1.0) for i in range(2)], dtype=torch.float)
    class_weights = compute_class_weights(labels_train).to(device)
    print("Computed class weights (inverse frequency):", class_weights)

    # Create LSTM model
    # Our model expects input of shape [B, T, C] where C=num_input_channels.
    model = AKI_LSTMClassifier(input_size=num_input_channels, hidden_size=128,
                               num_layers=2, num_classes=2, dropout=0.3).to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size, 16),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        report_to=["wandb"],
        no_cuda=not torch.cuda.is_available() or args.no_cuda,
        dataloader_num_workers=args.num_workers,
    )

    # Custom trainer for LSTM
    class LSTMTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = None

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            print("[DEBUG] Batch keys before loss computation:", list(inputs.keys()))
            labels = inputs.pop("labels").long()
            outputs = model(inputs["past_values"])
            if self.class_weights is not None:
                loss = F.cross_entropy(outputs, labels, weight=self.class_weights.to(labels.device))
            else:
                loss = F.cross_entropy(outputs, labels)
            return (loss, outputs) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                labels = inputs.pop("labels").long()
                outputs = model(inputs["past_values"])
                loss = F.cross_entropy(outputs, labels, weight=self.class_weights.to(labels.device) if self.class_weights is not None else None)
                return (loss, outputs, labels)

    trainer = LSTMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda eval_pred: {
            "accuracy": float(accuracy_score(eval_pred[1].argmax(axis=-1), eval_pred[2])),
            "auc": float(roc_auc_score(eval_pred[2], F.softmax(eval_pred[1], dim=-1)[:, 1])),
        },
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        data_collator=custom_data_collator,
    )
    trainer.class_weights = class_weights
    print("Using class weights:", class_weights)

    if args.no_save:
        print("Saving is disabled")
        training_args.save_steps = int(1e9)
        training_args.save_total_limit = 0

    print("\nStarting training...")
    trainer.train()

    if not args.no_save:
        try:
            model_path = os.path.join(args.output_dir, "final_model")
            trainer.save_model(model_path)
            print(f"Model saved successfully to {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    print("\nEvaluating final model...")
    results = trainer.evaluate()
    print("Final evaluation results:", results)

    return model, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="time_series_data_LSTM_10_29_2024",
                        help="Folder with per-patient CSV files.")
    parser.add_argument("--preprocessed_path", type=str, default="preprocessed_data.parquet",
                        help="Path to save/load preprocessed data.")
    parser.add_argument("--process_mode", type=str, choices=["truncate", "pool", "none"], default="pool",
                        help="Preprocessing mode.")
    parser.add_argument("--pool_window", type=int, default=60,
                        help="Window size for pooling (seconds).")
    parser.add_argument("--pool_method", type=str, choices=["average", "max", "median"], default="average",
                        help="Pooling method if using pool mode.")
    parser.add_argument("--fixed_length", type=int, default=10800,
                        help="Fixed length for truncate mode.")
    # Model parameters
    parser.add_argument("--patch_length", type=int, default=16,
                        help="(Unused for LSTM) Placeholder.")
    parser.add_argument("--patch_stride", type=int, default=8,
                        help="(Unused for LSTM) Placeholder.")
    parser.add_argument("--d_model", type=int, default=64,
                        help="(Unused for LSTM) Placeholder.")
    parser.add_argument("--num_hidden_layers", type=int, default=2,
                        help="(Unused for LSTM) Placeholder.")
    parser.add_argument("--num_attention_heads", type=int, default=8,
                        help="(Unused for LSTM) Placeholder.")
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./lstm_checkpoints")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for DataLoader.")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    # Hardware & debug
    parser.add_argument("--cuda", type=int, default=0, help="GPU device index to use.")
    parser.add_argument("--no_cuda", action="store_true", help="Do not use CUDA even if available.")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process fewer patients and print extra info.")
    parser.add_argument("--max_patients", type=int, default=10, help="Max patients to process in debug mode.")
    parser.add_argument("--no_save", action="store_true", help="Disable model saving.")

    args = parser.parse_args()
    main(args)
