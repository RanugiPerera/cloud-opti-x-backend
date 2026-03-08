import sys
from pathlib import Path
import logging
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.transformer import CostForecaster

# =============================================================================
# CONFIGURATION
# =============================================================================

PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
MODELS_DIR    = BASE_DIR / 'trained_models'
LOGS_DIR      = BASE_DIR / 'logs'

MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

SEQUENCE_LENGTH = 96        # 4 days of hourly history
FORECAST_LENGTH = 24        # predict next 24 hours
BATCH_SIZE      = 16        # smaller batch = better generalisation on small dataset
EPOCHS          = 150       # high ceiling — early stopping will trigger before this
LEARNING_RATE   = 0.0003
PATIENCE        = 20        # stop if val loss doesn't improve for 20 epochs
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'

FEATURE_COLS = [
    'cpu_usage', 'memory_usage', 'storage_gb', 'network_gb',
    'duration_hours', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
]
TARGET_COL = 'cost'

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = LOGS_DIR / f'training_{timestamp}.log'

    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                      datefmt='%H:%M:%S'))
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Config: seq={SEQUENCE_LENGTH}, forecast={FORECAST_LENGTH}, "
                f"batch={BATCH_SIZE}, lr={LEARNING_RATE}, patience={PATIENCE}")
    return logger


# =============================================================================
# DATASET
# =============================================================================

class CostTimeSeriesDataset(Dataset):
    def __init__(self, df, seq_length, forecast_length, logger,
                 feature_scaler=None, cost_scaler=None):
        self.seq_length      = seq_length
        self.forecast_length = forecast_length

        df = df.sort_values('hour').reset_index(drop=True)

        features = df[FEATURE_COLS].values
        cost     = df[TARGET_COL].values.reshape(-1, 1)

        # FIX: val set reuses train scalers to prevent data leakage
        if feature_scaler is None:
            self.feature_scaler = MinMaxScaler()
            self.feature_scaler.fit(features)
        else:
            self.feature_scaler = feature_scaler

        if cost_scaler is None:
            self.cost_scaler = MinMaxScaler()
            self.cost_scaler.fit(np.log1p(cost))
        else:
            self.cost_scaler = cost_scaler

        self.features = self.feature_scaler.transform(features)
        self.cost     = self.cost_scaler.transform(np.log1p(cost))

        if logger:
            logger.info("Dataset initialization:")
            logger.info(f"  Total records: {len(df)}")
            logger.info(f"  Feature count: {len(FEATURE_COLS)}")
            logger.info(f"  Cost range: ${cost.min():.4f} - ${cost.max():.4f}")

        self.sequences = self._create_sequences()

        if logger:
            logger.info(f"  Created {len(self.sequences)} sequences")

    def _create_sequences(self):
        sequences    = []
        total        = self.seq_length + self.forecast_length
        num_features = self.features.shape[1]

        for i in range(len(self.features) - total + 1):
            src = self.features[i : i + self.seq_length]

            # Decoder input: lagged cost in channel 0, zeros elsewhere
            cost_hist       = self.cost[
                i + self.seq_length - 1 : i + self.seq_length + self.forecast_length - 1
            ]
            tgt_input       = np.zeros((self.forecast_length, num_features))
            tgt_input[:, 0] = cost_hist[:, 0]

            # Target flattened to (forecast_len,) — prevents shape mismatch in loss
            tgt_output = self.cost[
                i + self.seq_length : i + self.seq_length + self.forecast_length
            ].flatten()

            sequences.append((src, tgt_input, tgt_output))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        src, tgt_input, tgt_output = self.sequences[idx]
        return (
            torch.tensor(src,        dtype=torch.float32),
            torch.tensor(tgt_input,  dtype=torch.float32),
            torch.tensor(tgt_output, dtype=torch.float32),
        )

    def inverse_transform_cost(self, values):
        values   = np.asarray(values).reshape(-1, 1)
        log_cost = self.cost_scaler.inverse_transform(values)
        return np.expm1(log_cost).flatten()


# =============================================================================
# TRAINING
# =============================================================================

def test_sample_prediction(model, val_loader, dataset, logger):
    model.model.eval()
    src, tgt_input, tgt_output = next(iter(val_loader))

    with torch.no_grad():
        preds  = model.model(src.to(model.device), tgt_input.to(model.device))
        pred   = preds[0, :, 0].cpu().numpy()
        actual = tgt_output[0].cpu().numpy()

    pred_d = dataset.inverse_transform_cost(pred)
    act_d  = dataset.inverse_transform_cost(actual)

    logger.info("  Sample prediction:")
    logger.info(f"    Predicted: avg=${pred_d.mean():.4f} "
                f"range=[${pred_d.min():.4f}, ${pred_d.max():.4f}]")
    logger.info(f"    Actual:    avg=${act_d.mean():.4f} "
                f"range=[${act_d.min():.4f}, ${act_d.max():.4f}]")


def train_model(model, train_loader, val_loader, dataset, logger):
    # HuberLoss is more robust to cost outliers than MSE
    criterion = nn.HuberLoss(delta=0.5)

    # AdamW has better weight decay regularisation than Adam
    optimizer = optim.AdamW(
        model.model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # Cosine annealing smoothly reduces LR to near-zero over training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    best_val         = float('inf')
    patience_counter = 0
    best_epoch       = 0
    train_losses, val_losses = [], []

    logger.info(f"\nStarting training — up to {EPOCHS} epochs, "
                f"early stopping patience={PATIENCE}")
    logger.info("=" * 60)

    for epoch in range(EPOCHS):

        # ---- Train ----
        model.model.train()
        train_loss = 0.0

        for src, tgt_input, tgt_output in train_loader:
            src        = src.to(model.device)
            tgt_input  = tgt_input.to(model.device)
            tgt_output = tgt_output.to(model.device)   # (batch, forecast_len)

            preds = model.model(src, tgt_input)         # (batch, forecast_len, 1)
            loss  = criterion(preds.squeeze(-1), tgt_output)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # ---- Validate ----
        model.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for src, tgt_input, tgt_output in val_loader:
                src        = src.to(model.device)
                tgt_input  = tgt_input.to(model.device)
                tgt_output = tgt_output.to(model.device)

                preds     = model.model(src, tgt_input)
                val_loss += criterion(preds.squeeze(-1), tgt_output).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step()

        # ---- Logging ----
        if (epoch + 1) % 5 == 0:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(f"Epoch [{epoch+1:3d}/{EPOCHS}] "
                        f"Train: {train_loss:.6f} | "
                        f"Val: {val_loss:.6f} | "
                        f"LR: {current_lr:.6f}")

        # ---- Checkpoint + early stopping ----
        if val_loss < best_val:
            best_val         = val_loss
            best_epoch       = epoch + 1
            patience_counter = 0
            model.save(MODELS_DIR / 'transformer_model.pth')

            if (epoch + 1) % 10 == 0:
                test_sample_prediction(model, val_loader, dataset, logger)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"\nEarly stopping at epoch {epoch + 1}")
                logger.info(f"Best val loss: {best_val:.6f} at epoch {best_epoch}")
                break

    logger.info(f"\nTraining complete.")
    logger.info(f"Best model: epoch {best_epoch}, val loss {best_val:.6f}")
    return train_losses, val_losses, best_epoch


# =============================================================================
# METRICS
# =============================================================================

def calculate_accuracy(model, val_loader, dataset, logger):
    model.load(MODELS_DIR / 'transformer_model.pth')
    model.model.eval()

    preds, trues = [], []

    with torch.no_grad():
        for src, tgt_input, tgt_output in val_loader:
            p = model.model(src.to(model.device), tgt_input.to(model.device))
            preds.extend(p.squeeze(-1).cpu().numpy().flatten())
            trues.extend(tgt_output.cpu().numpy().flatten())

    y_pred = dataset.inverse_transform_cost(preds)
    y_true = dataset.inverse_transform_cost(trues)

    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    logger.info("=" * 60)
    logger.info("FINAL METRICS (best checkpoint, validation set)")
    logger.info("=" * 60)
    logger.info(f"R²:   {r2:.4f}  (1.0 = perfect)")
    logger.info(f"MAE:  ${mae:.4f}")
    logger.info(f"RMSE: ${rmse:.4f}")
    logger.info(f"MAPE: {mape:.2f}%")

    return y_true, y_pred, r2, mae, rmse


# =============================================================================
# PLOTS
# =============================================================================

def plot_results(train_losses, val_losses, y_true, y_pred, best_epoch, logger):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(train_losses, label='Train Loss', color='steelblue', linewidth=1.5)
    axes[0].plot(val_losses,   label='Val Loss',   color='coral',     linewidth=1.5)
    if best_epoch <= len(val_losses):
        axes[0].axvline(best_epoch - 1, color='green', linestyle='--',
                        label=f'Best (epoch {best_epoch})', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Huber Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Predicted vs actual scatter
    max_val = max(y_true.max(), y_pred.max())
    axes[1].scatter(y_true, y_pred, alpha=0.3, s=5, color='steelblue')
    axes[1].plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Perfect fit')
    axes[1].set_xlabel('Actual Cost ($)')
    axes[1].set_ylabel('Predicted Cost ($)')
    axes[1].set_title('Predicted vs Actual')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = MODELS_DIR / 'training_results.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved: {plot_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger = setup_logging()

    df = pd.read_csv(PROCESSED_DIR / 'cost_timeseries.csv')
    logger.info(f"Loaded {len(df):,} rows from cost_timeseries.csv")

    # Time-based split — NEVER shuffle time series data
    split    = int(0.8 * len(df))
    train_df = df.iloc[:split].reset_index(drop=True)
    val_df   = df.iloc[split:].reset_index(drop=True)
    logger.info(f"Train rows: {len(train_df):,} | Val rows: {len(val_df):,}")

    # Build datasets
    train_dataset = CostTimeSeriesDataset(
        train_df, SEQUENCE_LENGTH, FORECAST_LENGTH, logger)

    # Val reuses train scalers — critical to prevent data leakage
    val_dataset = CostTimeSeriesDataset(
        val_df, SEQUENCE_LENGTH, FORECAST_LENGTH, logger=None,
        feature_scaler=train_dataset.feature_scaler,
        cost_scaler=train_dataset.cost_scaler,
    )

    logger.info(f"Train sequences: {len(train_dataset):,}")
    logger.info(f"Val sequences:   {len(val_dataset):,}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(
        val_dataset,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Smaller model — appropriate for ~3000 training sequences
    model = CostForecaster(
        input_dim=len(FEATURE_COLS),
        d_model=48,
        nhead=4,               # must divide d_model evenly: 48 / 4 = 12 OK
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.2,           # increased from 0.1 to combat overfitting
        output_dim=1,
        device=DEVICE
    )

    total_params = sum(p.numel() for p in model.model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Train
    train_losses, val_losses, best_epoch = train_model(
        model, train_loader, val_loader, train_dataset, logger)

    # Save scalers (must match what inference uses)
    with open(MODELS_DIR / 'feature_scaler.pkl', 'wb') as f:
        pickle.dump(train_dataset.feature_scaler, f)
    with open(MODELS_DIR / 'cost_scaler.pkl', 'wb') as f:
        pickle.dump(train_dataset.cost_scaler, f)
    logger.info("Scalers saved.")

    # Evaluate on best checkpoint
    y_true, y_pred, r2, mae, rmse = calculate_accuracy(
        model, val_loader, train_dataset, logger)

    # Save plots
    plot_results(train_losses, val_losses, y_true, y_pred, best_epoch, logger)

    logger.info("\nDone.")


if __name__ == '__main__':
    main()