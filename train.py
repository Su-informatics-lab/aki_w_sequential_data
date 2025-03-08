#!/usr/bin/env python
"""
PatchTST Classification for AKI Prediction

A time series classification model that uses PatchTST architecture from HuggingFace
to predict Acute Kidney Injury (AKI) from multi-channel patient vital signs data.

The process:
1. Load and preprocess time series data from individual patient CSV files
2. Combine into a single dataset with proper time indexing
3. Create ForecastDFDataset with static AKI labels
4. Train a PatchTSTForClassification model to predict binary AKI status
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import wandb
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

# Import core libraries
from tsfm_public.toolkit.dataset import ForecastDFDataset
from transformers import (
    PatchTSTConfig,
    PatchTSTForClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.configuration_utils import PretrainedConfig

# Patch PretrainedConfig's to_dict for JSON serialization
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
    """Pool over non-overlapping windows using average (ignoring NaNs)."""
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
            pooled_row[col] = np.nanmean(window[col]) if len(valid_vals) > 0 else np.nan

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
    """Dataset for on-the-fly processing of patient CSV files."""
    def __init__(self, file_list, label_dict, process_mode="pool", pool_window=60, fixed_length=10800):
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
        # Set AKI label using label_dict
        df["Acute_kidney_injury"] = int(self.label_dict.get(patient_id, np.nan))

        if self.process_mode == "truncate":
            df = truncate_pad_series(df, fixed_length=self.fixed_length)
        elif self.process_mode == "pool":
            df = pool_minute(df, pool_window=self.pool_window)

        df["Acute_kidney_injury"] = df["Acute_kidney_injury"].astype(np.int64)
        return df

class AKITrainer(Trainer):
    """Custom trainer for AKI classification with enhanced label handling."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_device = next(model.parameters()).device

        # Handle missing labels
        if "labels" not in inputs:
            if "static_categorical_values" in inputs:
                inputs["labels"] = inputs["static_categorical_values"].long().to(model_device)
            else:
                batch_size = inputs["past_values"].size(0)
                inputs["labels"] = torch.zeros(batch_size, dtype=torch.long, device=model_device)

        # Ensure labels are on the correct device
        labels = inputs.pop("labels").long().to(model_device)
        outputs = model(**inputs)

        # Apply class weights if available
        if self.class_weights is not None:
            class_weights = self.class_weights.to(model_device)
            loss = F.cross_entropy(outputs.prediction_logits, labels, weight=class_weights)
        else:
            loss = F.cross_entropy(outputs.prediction_logits, labels)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        model_device = next(model.parameters()).device

        # Handle missing labels
        if "labels" not in inputs:
            if "static_categorical_values" in inputs:
                inputs["labels"] = inputs["static_categorical_values"].long().to(model_device)
            else:
                batch_size = inputs["past_values"].size(0)
                inputs["labels"] = torch.zeros(batch_size, dtype=torch.long, device=model_device)

        with torch.no_grad():
            labels = inputs.pop("labels").long().to(model_device)
            outputs = model(**inputs)

            # Use class weights for evaluation if available
            class_weights = self.class_weights.to(model_device) if self.class_weights is not None else None
            loss = F.cross_entropy(outputs.prediction_logits, labels, weight=class_weights)

            return (loss, outputs.prediction_logits, labels)

def compute_classification_metrics(eval_pred):
    """Calculate classification metrics from model predictions."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )

    try:
        # For binary classification
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

def main(args):
    """Main training function."""
    # Initialize wandb and device
    wandb.init(project="patchtst_aki", config=vars(args))
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Load AKI labels from Excel file
    df_labels = pd.read_excel("imputed_demo_data.xlsx")
    df_labels = df_labels[["ID", "Acute_kidney_injury"]].drop_duplicates().dropna(subset=["Acute_kidney_injury"])
    label_dict = {str(x).strip(): y for x, y in zip(df_labels["ID"], df_labels["Acute_kidney_injury"])}

    # Filter file list to include only files with valid patient IDs
    file_list = [
        os.path.join(args.data_dir, fname)
        for fname in os.listdir(args.data_dir)
        if fname.endswith(".csv") and fname.split('_')[0].strip() in label_dict
    ]
    if args.debug:
        file_list = file_list[:args.max_patients]

    # Load or create preprocessed data
    preprocessed_path = args.preprocessed_path
    if os.path.exists(preprocessed_path):
        print(f"Loading preprocessed data from {preprocessed_path}...")
        all_patients_df = pd.read_parquet(preprocessed_path)
    else:
        # Process patient files and save to parquet
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

    # Ensure an "id" column exists and filter valid IDs
    all_patients_df["id"] = all_patients_df["ID"]
    all_patients_df = all_patients_df[all_patients_df["ID"].isin(label_dict.keys())]

    # Split data by patient ID
    unique_ids = all_patients_df["ID"].unique()
    train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
    train_df = all_patients_df[all_patients_df["ID"].isin(train_ids)]
    val_df = all_patients_df[all_patients_df["ID"].isin(val_ids)]

    # Determine feature columns
    feature_cols = [col for col in all_patients_df.columns
                    if col not in {"ID", "Acute_kidney_injury", "time_idx"}
                    and not col.startswith("Unnamed")
                    and np.issubdtype(all_patients_df[col].dtype, np.number)]

    print(f"Detected {len(feature_cols)} feature channels: {feature_cols}")
    num_input_channels = len(feature_cols)

    # Get maximum sequence length
    history_length = train_df.groupby("ID").size().max()
    print(f"History length (number of rows per patient): {history_length}")

    # Create datasets with AKI as static categorical column
    train_dataset = ForecastDFDataset(
        data=train_df,
        id_columns=["ID"],
        timestamp_column="time_idx",
        target_columns=feature_cols,
        observable_columns=feature_cols,
        context_length=history_length,
        prediction_length=1,
        static_categorical_columns=["Acute_kidney_injury"],
    )

    val_dataset = ForecastDFDataset(
        data=val_df,
        id_columns=["ID"],
        timestamp_column="time_idx",
        target_columns=feature_cols,
        observable_columns=feature_cols,
        context_length=history_length,
        prediction_length=1,
        static_categorical_columns=["Acute_kidney_injury"],
    )

    # Debug dataset inspection
    if args.debug and len(train_dataset) > 0:
        sample = train_dataset[0]
        print("\n[DEBUG] DATASET INSPECTION:")
        print(f"Sample keys: {list(sample.keys())}")
        if "past_values" in sample:
            print(f"past_values shape: {sample['past_values'].shape}")
        if "static_categorical_values" in sample:
            print(f"static_categorical_values: {sample['static_categorical_values']}")

        # Create a batch of one sample to check collation
        batch = {}
        for key in sample:
            if isinstance(sample[key], torch.Tensor):
                batch[key] = sample[key].unsqueeze(0)
            else:
                batch[key] = [sample[key]]
        print(f"Batch test keys: {list(batch.keys())}")

    # Compute class weights for imbalanced data
    labels_train = train_df["Acute_kidney_injury"].values
    unique, counts = np.unique(labels_train, return_counts=True)
    print("Training label distribution:", dict(zip(unique, counts)))

    # Inverse frequency weighting
    weight_list = []
    for i in range(2):  # Binary classification
        freq = counts[unique == i]
        if freq.size > 0:
            weight_list.append(1.0 / freq[0])
        else:
            weight_list.append(0.0)
    class_weights = torch.tensor(weight_list, dtype=torch.float).to(device)
    print("Computed class weights (inverse frequency):", class_weights)

    # Create model configuration
    config = PatchTSTConfig(
        num_input_channels=num_input_channels,
        context_length=history_length,
        prediction_length=1,
        num_targets=2,
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        dropout=0.1,
        head_dropout=0.2,
    )

    # Create and move model to device
    model = PatchTSTForClassification(config).to(device)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=1e-5,
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # Use F1 instead of loss for imbalanced data
        greater_is_better=True,      # Higher F1 is better
        report_to=["wandb"],
        no_cuda=not torch.cuda.is_available() or args.no_cuda,
        label_names=["static_categorical_values"],  # Tell trainer where labels are
    )

    # Create custom trainer
    trainer = AKITrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_classification_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    # Set class weights for weighted loss
    trainer.class_weights = class_weights
    print("Using class weights:", class_weights)

    # Disable saving if requested
    if args.no_save:
        print("Saving is disabled")
        training_args.save_steps = int(1e9)
        training_args.save_total_limit = 0

    # Train model
    print("\nStarting training...")
    trainer.train()

    # Save final model
    if not args.no_save:
        try:
            model_path = os.path.join(args.output_dir, "final_model")
            trainer.save_model(model_path)
            print(f"Model saved successfully to {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    # Final evaluation
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
                        help="Length of the patches.")
    parser.add_argument("--patch_stride", type=int, default=8,
                        help="Stride between patches.")
    parser.add_argument("--d_model", type=int, default=64,
                        help="Embedding dimension.")
    parser.add_argument("--num_hidden_layers", type=int, default=2,
                        help="Number of transformer layers.")
    parser.add_argument("--num_attention_heads", type=int, default=8,
                        help="Number of attention heads.")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./patchtst_checkpoints")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for DataLoader.")

    # Hardware & debug options
    parser.add_argument("--cuda", type=int, default=0,
                        help="GPU device index to use.")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Do not use CUDA even if available.")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: process fewer patients and print extra info.")
    parser.add_argument("--max_patients", type=int, default=10,
                        help="Max patients to process in debug mode.")
    parser.add_argument("--no_save", action="store_true",
                        help="Disable model saving.")

    args = parser.parse_args()
    main(args)