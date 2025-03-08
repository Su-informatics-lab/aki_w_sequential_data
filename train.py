#!/usr/bin/env python
"""
PatchTST Classification for AKI Prediction

Overall Process:
+--------------------------------------------------------------+
|                      Raw CSV Files                           |
|  Each patient file contains a time series with columns:      |
|  Monitoring signals (26 channels). The patient ID is inferred  |
|  from the file name (e.g., "R94657_combined.csv" yields ID "R94657").|
+--------------------------------------------------------------+
              |
              | Preprocessing: Each CSV is processed (via pooling or
              | truncation) to a fixed-length window. An "ID" column and
              | a "time_idx" column are added. The static AKI label is
              | obtained via the file name and the Excel file.
              v
+--------------------------------------------------------------+
|         Combined Preprocessed DataFrame (Parquet)            |
| Columns: ID, Acute_kidney_injury, time_idx, F1, F2, ..., F26     |
+--------------------------------------------------------------+
              |
              | Filter: Drop rows whose ID is not in the label mapping.
              v
+--------------------------------------------------------------+
|     Filtered DataFrame with only valid patient IDs           |
+--------------------------------------------------------------+
              |
              | Ensure an "id" column exists (copy from "ID").
              v
+--------------------------------------------------------------+
|         DataFrame with "ID" copied to "id"                   |
+--------------------------------------------------------------+
              |
              | Split by unique patient IDs into train and validation sets
              v
+--------------------------------------------------------------+
|   ForecastDFDataset (from tsfm_public.toolkit.dataset)         |
|  Configured with:                                              |
|     id_columns = ["ID"]                                         |
|     timestamp_column = "time_idx"                               |
|     target_columns = observable_columns = [F1, F2, ..., F26]    |
|  (Only observable signals (26 channels) are used as input)       |
+--------------------------------------------------------------+
              |
              | Wrap with ClassificationDataset to add a static AKI label,
              | by matching the patient ID (from "id") to the label mapping.
              v
+--------------------------------------------------------------+
|       ClassificationDataset (Custom Wrapper)                 |
|  Each example is a dict containing:                          |
|    - past_values: tensor of shape [context_length, 26]         |
|    - labels: scalar (AKI label from imputed_demo_data.xlsx)      |
+--------------------------------------------------------------+
              |
              | DataLoader batches examples using a custom collator
              v
+--------------------------------------------------------------+
|            PatchTST Classification Model                     |
|  Input: [batch, context_length, 26] observable channels         |
|  Classification head produces prediction logits                |
+--------------------------------------------------------------+
              |
              | Loss computed using provided "labels"
              v
+--------------------------------------------------------------+
|          Training via CustomTrainer (with EarlyStopping)       |
+--------------------------------------------------------------+
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import wandb
import torch.nn.functional as F
import json
from sklearn.metrics import roc_auc_score

# Import ForecastDFDataset from the tsfm_public toolkit.
from tsfm_public.toolkit.dataset import ForecastDFDataset

# Import PatchTST classes and EarlyStoppingCallback.
from transformers import (
    PatchTSTConfig,
    PatchTSTForClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

# --- Helper Functions ---

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

def pool_minute(df, pool_window=60):
    """
    Pool over non-overlapping windows of size pool_window using average (ignoring NaNs).
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
            if len(valid_vals) == 0:
                pooled_row[col] = np.nan
            else:
                pooled_row[col] = np.nanmean(window[col])
        pooled_data.append(pooled_row)
    return pd.DataFrame(pooled_data)

def truncate_pad_series(df, fixed_length, pad_value=0):
    """
    Truncate or pad df to exactly fixed_length rows.
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

class OnTheFlyForecastDFDataset(Dataset):
    def __init__(self, file_list, label_dict, process_mode="pool", pool_window=60, fixed_length=10800):
        """
        file_list: list of CSV file paths.
        label_dict: dict mapping patient ID -> AKI label.
        process_mode: "pool", "truncate", or "none".
        """
        self.file_list = file_list
        self.label_dict = label_dict
        self.process_mode = process_mode
        self.pool_window = pool_window
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
        # Set AKI label using label_dict; if missing, np.nan is returned.
        df["Acute_kidney_injury"] = int(self.label_dict.get(patient_id, np.nan))
        if self.process_mode == "truncate":
            df = truncate_pad_series(df, fixed_length=self.fixed_length)
        elif self.process_mode == "pool":
            df = pool_minute(df, pool_window=self.pool_window)
        df["Acute_kidney_injury"] = df["Acute_kidney_injury"].astype(np.int64)
        return df

def collate_patient_batches(batch):
    """
    Collate a list of processed patient DataFrames into one DataFrame.
    """
    return pd.concat(batch, ignore_index=True)

# --- Custom Dataset Wrapper for Classification ---
class ClassificationDataset(Dataset):
    """
    A wrapper around ForecastDFDataset for classification.
    It adds a static label "labels" to each example based on the patient ID.
    """
    def __init__(self, base_dataset, label_mapping, debug=False):
        self.base_dataset = base_dataset
        self.label_mapping = {str(k).strip(): v for k, v in label_mapping.items()}
        self.debug = debug

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Convert the output of the base dataset to a mutable dictionary.
        example = dict(self.base_dataset[idx])
        # Try to obtain patient ID from "id" then "ID"
        patient_id = None
        if "id" in example:
            patient_id = example["id"]
        elif "ID" in example:
            patient_id = example["ID"]
        if patient_id is None:
            raise KeyError("No patient ID found in example.")
        if isinstance(patient_id, (tuple, list)):
            patient_id = str(patient_id[0]).strip()
        else:
            patient_id = str(patient_id).strip()
        if patient_id not in self.label_mapping:
            raise KeyError(f"Patient ID {patient_id} not found in label mapping.")
        example["labels"] = torch.tensor(self.label_mapping[patient_id], dtype=torch.long)
        if self.debug and idx == 0:
            print(f"[DEBUG] ClassificationDataset example keys: {list(example.keys())}")
            print(f"[DEBUG] Patient ID: {patient_id}, Label: {example['labels']}")
        return example


# --- Custom Data Collator ---
def custom_data_collator(features):
    """
    Custom collate function to preserve all keys including 'labels'.
    """
    # Directly extract and stack labels first
    if all('labels' in f for f in features):
        batch = {
            'labels': torch.tensor([f['labels'] for f in features], dtype=torch.long)}
    else:
        print("WARNING: Not all samples have 'labels' key")
        batch = {}

    # Process other keys
    all_keys = set()
    for f in features:
        all_keys.update(f.keys())

    for key in all_keys:
        if key == 'labels':
            continue  # Already handled above

        # Try to stack if possible
        try:
            values = [f.get(key) for f in features if key in f]
            if all(v is not None and isinstance(v, torch.Tensor) for v in values):
                batch[key] = torch.stack(values)
            else:
                batch[key] = values
        except Exception as e:
            print(f"[DEBUG] Could not stack key '{key}': {e}")
            batch[key] = [f.get(key) for f in features if key in f]

    print(f"[DEBUG] Collated batch keys: {list(batch.keys())}")
    return batch

# --- Custom Trainer Subclass ---
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        print("[DEBUG] Batch keys before loss computation:", list(inputs.keys()))
        if "labels" not in inputs:
            raise KeyError(
                "'labels' not found in inputs. Check your data collator and dataset implementation.")

        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        if self.class_weights is not None:
            loss = F.cross_entropy(outputs.prediction_logits, labels,
                                   weight=self.class_weights.to(labels.device))
        else:
            loss = F.cross_entropy(outputs.prediction_logits, labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            labels = inputs.pop("labels").long()
            outputs = model(**inputs)
            loss = F.cross_entropy(outputs.prediction_logits, labels, weight=self.class_weights.to(labels.device) if self.class_weights is not None else None)
            return (loss, outputs.prediction_logits, labels)

# --- Main function ---
def main(args):
    wandb.init(project="patchtst_aki", config=vars(args))

    # Load AKI labels from Excel and drop rows with NaN labels.
    df_labels = pd.read_excel("imputed_demo_data.xlsx")
    df_labels = df_labels[["ID", "Acute_kidney_injury"]].drop_duplicates().dropna(subset=["Acute_kidney_injury"])
    label_dict = {str(x).strip(): y for x, y in zip(df_labels["ID"], df_labels["Acute_kidney_injury"])}

    # Filter file list: include only CSV files whose patient ID is in label_dict.
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

    # Ensure an "id" column exists.
    all_patients_df["id"] = all_patients_df["ID"]

    # Filter out rows whose "ID" is not in the label mapping.
    all_patients_df = all_patients_df[all_patients_df["ID"].isin(label_dict.keys())]

    # Split data by patient ID.
    unique_ids = all_patients_df["ID"].unique()
    train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
    train_df = all_patients_df[all_patients_df["ID"].isin(train_ids)]
    val_df = all_patients_df[all_patients_df["ID"].isin(val_ids)]

    # Determine observable features: exclude ID, target, and time_idx.
    feature_cols = [col for col in all_patients_df.columns
                    if col not in {"ID", "Acute_kidney_injury", "time_idx"}
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
    )
    base_val_dataset = ForecastDFDataset(
        data=val_df,
        id_columns=["ID"],
        timestamp_column="time_idx",
        target_columns=feature_cols,
        observable_columns=feature_cols,
        context_length=history_length,
        prediction_length=1,
    )
    # Wrap these with ClassificationDataset to add the static AKI label.
    train_dataset = ClassificationDataset(base_train_dataset, label_dict, debug=args.debug)
    val_dataset = ClassificationDataset(base_val_dataset, label_dict, debug=args.debug)

    # (Optional) Print a sample from the training dataset for debugging.
    if args.debug and len(train_dataset) > 0:
        sample = train_dataset[0]
        print("[DEBUG] Sample training example keys:", list(sample.keys()))
        if "past_values" in sample:
            print("[DEBUG] past_values shape:", sample["past_values"].shape)
        print("[DEBUG] labels:", sample["labels"])

    # Create DataLoaders using our custom data collator.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=custom_data_collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_data_collator)

    # Compute class weights from training labels.
    labels_train = train_df["Acute_kidney_injury"].values
    unique, counts = np.unique(labels_train, return_counts=True)
    print("Training label distribution:", dict(zip(unique, counts)))
    num_classes = 2
    weight_list = []
    for i in range(num_classes):
        freq = counts[unique == i]
        if freq.size > 0:
            weight_list.append(1.0 / freq[0])
        else:
            weight_list.append(0.0)
    class_weights = torch.tensor(weight_list, dtype=torch.float)
    print("Computed class weights (inverse frequency):", class_weights)

    # Create PatchTST configuration using only observable features.
    config_dict = {
        "num_input_channels": int(num_input_channels),  # 26 channels
        "context_length": int(history_length),
        "prediction_length": 1,
        "num_targets": 2,
        "patch_length": int(args.patch_length),
        "patch_stride": int(args.patch_stride),
        "d_model": int(args.d_model),
        "num_hidden_layers": int(args.num_hidden_layers),
        "num_attention_heads": int(args.num_attention_heads),
    }
    config = PatchTSTConfig(**config_dict)
    model = PatchTSTForClassification(config)

    # Create training arguments with a lower learning rate.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=1e-6,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        report_to=["wandb"],
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        accuracy = (preds == labels).mean()
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        try:
            auc = roc_auc_score(labels, preds)
        except Exception:
            auc = 0.0
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc)
        }

    # Create our custom trainer with EarlyStoppingCallback.
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        data_collator=custom_data_collator,
    )
    trainer.class_weights = class_weights
    print("Using class weights:", class_weights)

    if args.no_save:
        print("Saving is disabled")
        training_args.save_steps = int(1e9)
        training_args.save_total_limit = 0

    trainer.train()

    if not args.no_save:
        try:
            trainer.save_model(os.path.join(args.output_dir, "final_model"))
            print(f"Model saved successfully to {os.path.join(args.output_dir, 'final_model')}")
        except Exception as e:
            print(f"Error saving model: {e}")

    trainer.evaluate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="time_series_data_LSTM_10_29_2024",
                        help="Folder with per-patient CSV files.")
    parser.add_argument("--preprocessed_path", type=str, default="preprocessed_data.parquet",
                        help="Path to save/load preprocessed data (Parquet format).")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process only a few patients.")
    parser.add_argument("--max_patients", type=int, default=10, help="Max patients to process in debug mode.")
    parser.add_argument("--process_mode", type=str, choices=["truncate", "pool", "none"], default="pool",
                        help="Preprocessing mode: 'truncate' to pad/truncate, 'pool' for minute pooling, or 'none'.")
    parser.add_argument("--fixed_length", type=int, default=10800,
                        help="Fixed length if using 'truncate' mode (e.g., 10800 for 3 hours at 1-second resolution).")
    parser.add_argument("--pool_window", type=int, default=60,
                        help="Window size for pooling (e.g., 60 seconds).")
    parser.add_argument("--pool_method", type=str, choices=["average", "max", "median"], default="average",
                        help="Pooling method if using pool mode.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patch_length", type=int, default=16)
    parser.add_argument("--patch_stride", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./patchtst_checkpoints")
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for DataLoader.")
    parser.add_argument("--no_save", action="store_true", help="Disable model saving.")

    args = parser.parse_args()
    main(args)
