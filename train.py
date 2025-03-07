#!/usr/bin/env python
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

# Import your ForecastDFDataset from utils.
from utils import ForecastDFDataset

# Import PatchTST classes from Hugging Face Transformers.
from transformers import PatchTSTConfig, PatchTSTForClassification, Trainer, TrainingArguments

# --- Helper Functions ---

def pool_minute(df, pool_window=60):
    """
    Given a patient's DataFrame (each row is one second, columns are measures),
    pool over non-overlapping windows of size pool_window (in seconds) using average,
    ignoring NaNs. Returns a new DataFrame with pooled rows.
    """
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
            if len(valid_vals) == 0:
                pooled_row[col] = np.nan
            else:
                pooled_row[col] = np.nanmean(window[col])
        pooled_data.append(pooled_row)
    return pd.DataFrame(pooled_data)

def truncate_pad_series(df, fixed_length, pad_value=0):
    """
    Truncate df if its length is greater than fixed_length; if shorter, pad with pad_value.
    Returns a DataFrame with exactly fixed_length rows.
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
        file_list: list of CSV file paths (one per patient)
        label_dict: dictionary mapping patient ID -> AKI label
        process_mode: "pool", "truncate", or "none"
        pool_window: window (in seconds) for pooling
        fixed_length: fixed length (in seconds) if using "truncate" mode.
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
        patient_id = fname.split('_')[0]
        df = pd.read_csv(csv_path)
        df["time_idx"] = range(len(df))
        df["ID"] = patient_id
        # Cast label to int and drop NaNs by default from label_dict
        df["Acute_kidney_injury"] = int(self.label_dict.get(patient_id, 0))
        if self.process_mode == "truncate":
            df = truncate_pad_series(df, fixed_length=self.fixed_length)
        elif self.process_mode == "pool":
            df = pool_minute(df, pool_window=self.pool_window)
        df["Acute_kidney_injury"] = df["Acute_kidney_injury"].astype(np.int64)
        return df

def collate_patient_batches(batch):
    """
    Given a list of processed patient DataFrames (one per patient), collate them into one DataFrame.
    """
    return pd.concat(batch, ignore_index=True)

# --- Main function ---
def main(args):
    wandb.init(project="patchtst_aki", config=vars(args))

    # Load AKI labels from Excel and drop rows with NaN in the label column.
    df_labels = pd.read_excel("imputed_demo_data.xlsx")
    df_labels = df_labels[["ID", "Acute_kidney_injury"]].drop_duplicates().dropna(subset=["Acute_kidney_injury"])
    label_dict = dict(zip(df_labels["ID"], df_labels["Acute_kidney_injury"]))

    preprocessed_path = args.preprocessed_path
    if os.path.exists(preprocessed_path):
        print(f"Loading preprocessed data from {preprocessed_path}...")
        all_patients_df = pd.read_parquet(preprocessed_path)
    else:
        file_list = [os.path.join(args.data_dir, fname) for fname in os.listdir(args.data_dir) if fname.endswith(".csv")]
        if args.debug:
            file_list = file_list[:args.max_patients]

        dataset = OnTheFlyForecastDFDataset(
            file_list=file_list,
            label_dict=label_dict,
            process_mode=args.process_mode,
            pool_window=args.pool_window,
            fixed_length=args.fixed_length
        )

        print("Processing patient files...")
        writer = None
        for i in tqdm(range(len(dataset))):
            df = dataset[i]
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

    # Split by patient ID so that each patient's series remains intact.
    unique_ids = all_patients_df["ID"].unique()
    train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
    train_df = all_patients_df[all_patients_df["ID"].isin(train_ids)]
    val_df = all_patients_df[all_patients_df["ID"].isin(val_ids)]

    # Determine feature columns: numeric columns except ID, Acute_kidney_injury, time_idx.
    feature_cols = [col for col in all_patients_df.columns if col not in {"ID", "Acute_kidney_injury", "time_idx"}
                    and np.issubdtype(all_patients_df[col].dtype, np.number)]
    # Remove any extra columns that start with "Unnamed"
    feature_cols = [col for col in feature_cols if not col.startswith("Unnamed")]
    print(f"Detected {len(feature_cols)} feature channels: {feature_cols}")

    history_length = train_df.groupby("ID").size().max()

    # Create ForecastDFDataset objects using the proper parameter names.
    train_dataset = ForecastDFDataset(
        data=train_df,
        id_columns=["ID"],
        timestamp_column="time_idx",
        target_columns=["Acute_kidney_injury"],
        observable_columns=feature_cols,
        context_length=history_length,
        prediction_length=1,
    )
    val_dataset = ForecastDFDataset(
        data=val_df,
        id_columns=["ID"],
        timestamp_column="time_idx",
        target_columns=["Acute_kidney_injury"],
        observable_columns=feature_cols,
        context_length=history_length,
        prediction_length=1,
    )

    # Wrap datasets with torch DataLoader.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Configure PatchTST model for classification.
    config = PatchTSTConfig(
        num_input_channels=len(feature_cols),
        context_length=history_length,
        prediction_length=1,
        num_targets=2,  # binary classification: 0 and 1
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
    )
    model = PatchTSTForClassification(config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
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
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
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

    args = parser.parse_args()
    main(args)
