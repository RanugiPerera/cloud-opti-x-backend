import sys
from pathlib import Path

# add parent directory to path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from models.transformer import CostForecaster

# config
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'trained_models'
MODELS_DIR.mkdir(exist_ok=True)

# hyperparameters
SEQUENCE_LENGTH = 24  # use past 24 hours to predict
FORECAST_LENGTH = 7  # predict next 7 hours
BATCH_SIZE = 32
EPOCHS = 100  # increased from 50
LEARNING_RATE = 0.0005  # reduced for stability
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print("=" * 70)
print("TRANSFORMER MODEL TRAINING - CLOUD COST FORECASTING")
print("=" * 70)
print(f"Device: {DEVICE}")
print(f"Sequence length: {SEQUENCE_LENGTH} hours")
print(f"Forecast length: {FORECAST_LENGTH} hours")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print("=" * 70)


# ========================================
# Dataset Class (UPDATED WITH 6 FEATURES)
# ========================================

class CostTimeSeriesDataset(Dataset):
    """
    Dataset for time series cost data
    Creates sequences for training
    """

    def __init__(self, data, seq_length=24, forecast_length=7):
        self.seq_length = seq_length
        self.forecast_length = forecast_length

        # features to use - NOW INCLUDES STORAGE AND NETWORK
        self.feature_cols = ['cpu_usage', 'memory_usage', 'storage_gb', 'network_gb', 'duration_hours', 'cost']

        # sort by time
        self.data = data.sort_values('hour').reset_index(drop=True)

        print(f"\nDataset info:")
        print(f"  Total records: {len(self.data)}")
        print(f"  Features: {self.feature_cols}")

        # extract and normalize features
        features = self.data[self.feature_cols].values

        # use MinMaxScaler for [0, 1] range (prevents negative values)
        self.scaler = MinMaxScaler()
        self.features_normalized = self.scaler.fit_transform(features)

        print(f"  Original cost range: ${features[:, -1].min():.4f} - ${features[:, -1].max():.4f}")
        print(f"  Normalized cost range: {self.features_normalized[:, -1].min():.4f} - {self.features_normalized[:, -1].max():.4f}")

        # create sequences
        self.sequences = self._create_sequences()

        print(f"  Created {len(self.sequences)} sequences")

    def _create_sequences(self):
        """Create input-output sequence pairs"""
        sequences = []

        total_length = self.seq_length + self.forecast_length

        for i in range(len(self.features_normalized) - total_length + 1):
            # input: past seq_length hours
            src = self.features_normalized[i:i + self.seq_length]

            # decoder input: seq_length-1 past + forecast_length future
            tgt_input = self.features_normalized[i + self.seq_length - 1:i + self.seq_length + self.forecast_length - 1]

            # output: next forecast_length hours (cost only)
            tgt_output = self.features_normalized[i + self.seq_length:i + self.seq_length + self.forecast_length, -1]

            sequences.append((src, tgt_input, tgt_output))

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        src, tgt_input, tgt_output = self.sequences[idx]

        return (
            torch.FloatTensor(src),
            torch.FloatTensor(tgt_input),
            torch.FloatTensor(tgt_output)
        )

    def inverse_transform_cost(self, normalized_costs):
        """Convert normalized costs back to original scale"""
        # create dummy array with same shape as original features
        dummy = np.zeros((len(normalized_costs), len(self.feature_cols)))
        dummy[:, -1] = normalized_costs  # put costs in last column

        # inverse transform
        original = self.scaler.inverse_transform(dummy)
        return original[:, -1]  # return only cost column


# ========================================
# Training Function (Improved)
# ========================================

def train_model(model, train_loader, val_loader, dataset, epochs=100, lr=0.0005):
    """
    Train the transformer model with improved training loop
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=10,
        factor=0.5
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20

    print("\nStarting training...")
    print("=" * 70)

    for epoch in range(epochs):
        # training
        model.model.train()
        train_loss = 0
        train_batches = 0

        for batch_idx, (src, tgt_input, tgt_output) in enumerate(train_loader):
            src = src.to(model.device)
            tgt_input = tgt_input.to(model.device)
            tgt_output = tgt_output.to(model.device)

            # forward pass
            predictions = model.model(src, tgt_input)

            # calculate loss
            loss = criterion(predictions.squeeze(-1), tgt_output)

            # backward pass
            optimizer.zero_grad()
            loss.backward()

            # gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        train_loss /= train_batches
        train_losses.append(train_loss)

        # validation
        model.model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for src, tgt_input, tgt_output in val_loader:
                src = src.to(model.device)
                tgt_input = tgt_input.to(model.device)
                tgt_output = tgt_output.to(model.device)

                predictions = model.model(src, tgt_input)
                loss = criterion(predictions.squeeze(-1), tgt_output)
                val_loss += loss.item()
                val_batches += 1

        val_loss /= val_batches
        val_losses.append(val_loss)

        # learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        # manually print if LR changed (since verbose is not available)
        if old_lr != new_lr:
            print(f"  → Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")

        # print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1:3d}/{epochs}] - "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(MODELS_DIR / 'transformer_model.pth')
            patience_counter = 0

            # test a sample prediction
            if (epoch + 1) % 10 == 0:
                test_sample_prediction(model, val_loader, dataset)
        else:
            patience_counter += 1

        # early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    print("=" * 70)
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return train_losses, val_losses


def test_sample_prediction(model, val_loader, dataset):
    """Test model with a sample to check if predictions are reasonable"""
    model.model.eval()

    # get one batch
    src, tgt_input, tgt_output = next(iter(val_loader))
    src = src.to(model.device)
    tgt_input = tgt_input.to(model.device)

    with torch.no_grad():
        predictions = model.model(src, tgt_input)
        pred_normalized = predictions[0, :, 0].cpu().numpy()
        actual_normalized = tgt_output[0].cpu().numpy()

        # denormalize
        pred_original = dataset.inverse_transform_cost(pred_normalized)
        actual_original = dataset.inverse_transform_cost(actual_normalized)

        print(f"\n  Sample prediction check:")
        print(f"    Predicted costs: ${pred_original.mean():.4f} (avg), ${pred_original.min():.4f}-${pred_original.max():.4f} (range)")
        print(f"    Actual costs:    ${actual_original.mean():.4f} (avg), ${actual_original.min():.4f}-${actual_original.max():.4f} (range)")


# ========================================
# Main Execution
# ========================================

if __name__ == '__main__':
    print("\n[1/5] Loading processed data...")

    # load cost time series
    cost_df = pd.read_csv(PROCESSED_DIR / 'cost_timeseries.csv')
    print(f"✓ Loaded {len(cost_df):,} records")

    # create dataset
    print("\n[2/5] Creating dataset...")
    dataset = CostTimeSeriesDataset(
        cost_df,
        seq_length=SEQUENCE_LENGTH,
        forecast_length=FORECAST_LENGTH
    )

    # split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )

    print(f"\n[3/5] Creating data loaders...")
    print(f"  Train: {len(train_dataset):,} sequences")
    print(f"  Val: {len(val_dataset):,} sequences")

    # create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # create model - NOW WITH 6 INPUT DIMENSIONS
    print("\n[4/5] Initializing model...")
    model = CostForecaster(
        input_dim=6,  # CHANGED FROM 4 TO 6
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        output_dim=1,
        device=DEVICE
    )

    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # train
    print("\n[5/5] Training model...")
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        dataset,
        epochs=EPOCHS,
        lr=LEARNING_RATE
    )

    # save scaler
    print("\nSaving scaler...")
    scaler_path = MODELS_DIR / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(dataset.scaler, f)
    print(f"✓ Scaler saved to {scaler_path}")

    # plot training curves
    print("\nGenerating training plots...")
    plt.figure(figsize=(12, 5))

    # loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(val_losses, label='Val Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # zoomed loss plot (last 50% of training)
    plt.subplot(1, 2, 2)
    start_idx = len(train_losses) // 2
    plt.plot(range(start_idx, len(train_losses)), train_losses[start_idx:], label='Train Loss', alpha=0.7)
    plt.plot(range(start_idx, len(val_losses)), val_losses[start_idx:], label='Val Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress (Last 50%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'training_curve.png', dpi=150)
    print(f"✓ Plot saved to: {MODELS_DIR / 'training_curve.png'}")

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Model saved to: {MODELS_DIR / 'transformer_model.pth'}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Training plot saved to: {MODELS_DIR / 'training_curve.png'}")
    print("\nYou can now use the model for forecasting!")
    print("Restart your Flask server to load the new model.")
    print("=" * 70)
    
    # ========================================
# Final Accuracy Evaluation
# ========================================
from sklearn.metrics import r2_score, mean_absolute_percentage_error

print("\n[6/5] Calculating Final Accuracy Metrics...")

# Load the best model saved during training
model.model.load_state_dict(torch.load(MODELS_DIR / 'transformer_model.pth'))
model.model.eval()

all_preds = []
all_actuals = []

with torch.no_grad():
    for src, tgt_input, tgt_output in val_loader:
        src = src.to(model.device)
        tgt_input = tgt_input.to(model.device)
        
        # Get model predictions
        predictions = model.model(src, tgt_input)
        
        # We only care about the cost (last dimension)
        # Flatten the batch and forecast length
        pred_flat = predictions.squeeze(-1).cpu().numpy().flatten()
        actual_flat = tgt_output.cpu().numpy().flatten()
        
        all_preds.extend(pred_flat)
        all_actuals.extend(actual_flat)

# Convert to original dollar scale using the dataset's inverse transform
# Note: inverse_transform_cost expects a 1D array
y_pred_dollars = dataset.inverse_transform_cost(np.array(all_preds))
y_true_dollars = dataset.inverse_transform_cost(np.array(all_actuals))

# Calculate Accuracy Metrics
r2 = r2_score(y_true_dollars, y_pred_dollars)
mape = mean_absolute_percentage_error(y_true_dollars, y_pred_dollars)
accuracy_pct = 100 * (1 - mape)

print("-" * 30)
print(f"Final R2 Score (Variance Explained): {r2:.4f}")
print(f"Mean Absolute Percentage Error:    {mape*100:.2f}%")
print(f"OVERALL MODEL ACCURACY:            {accuracy_pct:.2f}%")
print("-" * 30)

# Optional: Save a Scatter Plot of Actual vs Predicted
plt.figure(figsize=(7, 7))
plt.scatter(y_true_dollars, y_pred_dollars, alpha=0.3, color='teal', s=10)
# Add identity line
max_val = max(y_true_dollars.max(), y_pred_dollars.max())
plt.plot([0, max_val], [0, max_val], color='red', lw=2, linestyle='--')
plt.xlabel('Actual Cost ($)')
plt.ylabel('Predicted Cost ($)')
plt.title(f'Actual vs Predicted (Accuracy: {accuracy_pct:.2f}%)')
plt.grid(True, alpha=0.2)
plt.savefig(MODELS_DIR / 'accuracy_scatter.png')
print(f"✓ Accuracy scatter plot saved to: {MODELS_DIR / 'accuracy_scatter.png'}")