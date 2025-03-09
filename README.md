# AKI Prediction with Recurrent Neural Networks

This repository provides a comprehensive framework for predicting Acute Kidney Injury (AKI) using recurrent neural network architectures (LSTM and GRU) on multi-channel patient vital signs data.

## Overview

The project encompasses the following key functionalities:

- **Data Preprocessing:**  
  Raw CSV files (one per patient) are processed to generate time-indexed sequences. This includes optional pooling (e.g., averaging over 60-second intervals), truncating/padding to a fixed sequence length, imputing missing values, and normalizing features based on training set statistics.

- **Model Training:**  
  The framework supports training recurrent classifiers (LSTM or GRU) with an optional attention mechanism. Training employs early stopping and utilizes Weights & Biases (wandb) for experiment tracking and logging.

- **Evaluation and Checkpointing:**  
  The system computes performance metrics (accuracy, AUC, F1-score) and saves model checkpoints with descriptive run names that reflect the chosen architecture and hyperparameters, facilitating easy model recovery.

## Installation

Ensure you have Python 3.10 or later installed. Install the necessary dependencies by running:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Usage

### Running the Training Script

Execute the training script with appropriate command-line arguments. For example:

```bash
python train.py  # lstm w/t attn
```

### Command-Line Options

- **--gru**  
  Use GRU instead of LSTM.

- **--attention**  
  Enable the attention mechanism on top of the recurrent layer.

- **--batch_size, --learning_rate, --epochs, etc.**  
  Customize hyperparameters as needed. For a full list of options, run:

```bash
python train.py --help
```

## Contact

For further inquiries or contributions, please contact [hw56@iu.edu](mailto:hw56@iu.edu).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for further details.
