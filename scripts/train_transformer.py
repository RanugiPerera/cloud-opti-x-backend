"""
Transformer Model Training - Cloud Cost Forecasting
"""

import sys
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path
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
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

from models.transformer import CostForecaster

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'trained_models'
LOGS_DIR = BASE_DIR / 'logs'

# Create directories
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Hyperparameters
SEQUENCE_LENGTH = 24  # use past 24 hours to predict
FORECAST_LENGTH = 24  # predict next 24 hours
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0005
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Feature columns
FEATURE_COLS = ['cpu_usage', 'memory_usage', 'storage_gb', 'network_gb', 'duration_hours', 'cost']


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging() -> logging.Logger:
    """Configure logging with both file and console handlers"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOGS_DIR / f'training_{timestamp}.log'
    
    # Create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # Try to set UTF-8 encoding for console (Windows fix)
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler with explicit UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


# ============================================================================
# DATASET CLASS
# ============================================================================

class CostTimeSeriesDataset(Dataset):
    """
    Dataset for time series cost data
    Creates sequences for training
    """

    def __init__(self, data, seq_length=24, forecast_length=7, logger=None):
        self.seq_length = seq_length
        self.forecast_length = forecast_length
        self.logger = logger or logging.getLogger('training')

        # Features to use
        self.feature_cols = FEATURE_COLS

        # Sort by time
        self.data = data.sort_values('hour').reset_index(drop=True)

        self.logger.info(f"Dataset initialization:")
        self.logger.info(f"  Total records: {len(self.data):,}")
        self.logger.info(f"  Features: {self.feature_cols}")

        # Extract and normalize features
        features = self.data[self.feature_cols].values

        # Use MinMaxScaler for [0, 1] range
        self.scaler = MinMaxScaler()
        self.features_normalized = self.scaler.fit_transform(features)

        self.logger.info(f"  Original cost range: ${features[:, -1].min():.4f} - ${features[:, -1].max():.4f}")
        self.logger.info(f"  Normalized cost range: {self.features_normalized[:, -1].min():.4f} - {self.features_normalized[:, -1].max():.4f}")

        # Create sequences
        self.sequences = self._create_sequences()

        self.logger.info(f"  Created {len(self.sequences):,} sequences")

    def _create_sequences(self):
        """Create input-output sequence pairs"""
        sequences = []
        total_length = self.seq_length + self.forecast_length

        for i in range(len(self.features_normalized) - total_length + 1):
            # Input: past seq_length hours
            src = self.features_normalized[i:i + self.seq_length]

            # Decoder input: seq_length-1 past + forecast_length future
            tgt_input = self.features_normalized[i + self.seq_length - 1:i + self.seq_length + self.forecast_length - 1]

            # Output: next forecast_length hours (cost only)
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
        dummy = np.zeros((len(normalized_costs), len(self.feature_cols)))
        dummy[:, -1] = normalized_costs
        original = self.scaler.inverse_transform(dummy)
        return original[:, -1]


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def test_sample_prediction(model, val_loader, dataset, logger):
    """Test model with a sample to check if predictions are reasonable"""
    model.model.eval()

    # Get one batch
    src, tgt_input, tgt_output = next(iter(val_loader))
    src = src.to(model.device)
    tgt_input = tgt_input.to(model.device)

    with torch.no_grad():
        predictions = model.model(src, tgt_input)
        pred_normalized = predictions[0, :, 0].cpu().numpy()
        actual_normalized = tgt_output[0].cpu().numpy()

        # Denormalize
        pred_original = dataset.inverse_transform_cost(pred_normalized)
        actual_original = dataset.inverse_transform_cost(actual_normalized)

        logger.info(f"  Sample prediction check:")
        logger.info(f"    Predicted costs: ${pred_original.mean():.4f} (avg), ${pred_original.min():.4f}-${pred_original.max():.4f} (range)")
        logger.info(f"    Actual costs:    ${actual_original.mean():.4f} (avg), ${actual_original.min():.4f}-${actual_original.max():.4f} (range)")


def train_model(model, train_loader, val_loader, dataset, logger, epochs=100, lr=0.0005):
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

    logger.info("="*70)
    logger.info("Starting training...")
    logger.info("="*70)

    for epoch in range(epochs):
        # Training
        model.model.train()
        train_loss = 0
        train_batches = 0

        for batch_idx, (src, tgt_input, tgt_output) in enumerate(train_loader):
            src = src.to(model.device)
            tgt_input = tgt_input.to(model.device)
            tgt_output = tgt_output.to(model.device)

            # Forward pass
            predictions = model.model(src, tgt_input)

            # Calculate loss
            loss = criterion(predictions.squeeze(-1), tgt_output)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        train_loss /= train_batches
        train_losses.append(train_loss)

        # Validation
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

        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if old_lr != new_lr:
            logger.info(f"  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(f"Epoch [{epoch + 1:3d}/{epochs}] - "
                       f"Train Loss: {train_loss:.6f}, "
                       f"Val Loss: {val_loss:.6f}, "
                       f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(MODELS_DIR / 'transformer_model.pth')
            patience_counter = 0

            # Test a sample prediction
            if (epoch + 1) % 10 == 0:
                test_sample_prediction(model, val_loader, dataset, logger)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stop_patience:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    logger.info("="*70)
    logger.info(f"Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")

    return train_losses, val_losses


def calculate_accuracy(model, val_loader, dataset, logger):
    """Calculate final accuracy metrics"""
    
    logger.info("="*70)
    logger.info("Calculating Final Accuracy Metrics...")
    logger.info("="*70)

    # Load best model
    model.model.load_state_dict(torch.load(MODELS_DIR / 'transformer_model.pth'))
    model.model.eval()

    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for src, tgt_input, tgt_output in val_loader:
            src = src.to(model.device)
            tgt_input = tgt_input.to(model.device)

            # Get predictions
            predictions = model.model(src, tgt_input)

            # Flatten
            pred_flat = predictions.squeeze(-1).cpu().numpy().flatten()
            actual_flat = tgt_output.cpu().numpy().flatten()

            all_preds.extend(pred_flat)
            all_actuals.extend(actual_flat)

    # Convert to original dollar scale
    y_pred_dollars = dataset.inverse_transform_cost(np.array(all_preds))
    y_true_dollars = dataset.inverse_transform_cost(np.array(all_actuals))

    # Calculate metrics
    r2 = r2_score(y_true_dollars, y_pred_dollars)
    mape = mean_absolute_percentage_error(y_true_dollars, y_pred_dollars)
    accuracy_pct = 100 * (1 - mape)

    logger.info("-" * 70)
    logger.info(f"R2 Score (Variance Explained):     {r2:.4f}")
    logger.info(f"Mean Absolute Percentage Error:    {mape*100:.2f}%")
    logger.info(f"OVERALL MODEL ACCURACY:            {accuracy_pct:.2f}%")
    logger.info("-" * 70)

    return y_true_dollars, y_pred_dollars, accuracy_pct


def plot_training_curves(train_losses, val_losses, logger):
    """Generate and save training curve plots"""
    
    logger.info("Generating training plots...")
    
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(val_losses, label='Val Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Zoomed loss plot
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
    plot_path = MODELS_DIR / 'training_curve.png'
    plt.savefig(plot_path, dpi=150)
    logger.info(f"[OK] Training curve saved to: {plot_path}")
    plt.close()


def plot_accuracy_scatter(y_true, y_pred, accuracy_pct, logger):
    """Generate and save accuracy scatter plot"""
    
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.3, color='teal', s=10)
    
    # Add identity line
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], color='red', lw=2, linestyle='--')
    
    plt.xlabel('Actual Cost ($)')
    plt.ylabel('Predicted Cost ($)')
    plt.title(f'Actual vs Predicted (Accuracy: {accuracy_pct:.2f}%)')
    plt.grid(True, alpha=0.2)
    
    plot_path = MODELS_DIR / 'accuracy_scatter.png'
    plt.savefig(plot_path, dpi=150)
    logger.info(f"[OK] Accuracy scatter plot saved to: {plot_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    
    # Setup logging
    logger = setup_logging()
    
    try:
        logger.info("="*70)
        logger.info("TRANSFORMER MODEL TRAINING - CLOUD COST FORECASTING")
        logger.info("="*70)
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Sequence length: {SEQUENCE_LENGTH} hours")
        logger.info(f"Forecast length: {FORECAST_LENGTH} hours")
        logger.info(f"Batch size: {BATCH_SIZE}")
        logger.info(f"Epochs: {EPOCHS}")
        logger.info(f"Learning rate: {LEARNING_RATE}")
        logger.info("="*70)

        # Step 1: Load data
        logger.info("\n[1/6] Loading processed data...")
        cost_df = pd.read_csv(PROCESSED_DIR / 'cost_timeseries.csv')
        logger.info(f"[OK] Loaded {len(cost_df):,} records")
        
            # Step 1: Load data
        logger.info("\n[1/6] Loading processed data...")
        cost_df = pd.read_csv(PROCESSED_DIR / 'cost_timeseries.csv')
        logger.info(f"[OK] Loaded {len(cost_df):,} records")

        # ------------------------------------------------------------------
        # DATA SUFFICIENCY CHECK
        # ------------------------------------------------------------------
        min_required = SEQUENCE_LENGTH + FORECAST_LENGTH

        logger.info("Data requirements check:")
        logger.info(f"  Sequence length: {SEQUENCE_LENGTH}")
        logger.info(f"  Forecast length: {FORECAST_LENGTH}")
        logger.info(f"  Total required hours: {min_required}")
        logger.info(f"  Available data hours: {len(cost_df)}")

        if len(cost_df) < min_required:
            logger.error(
                f"❌ Insufficient data: need at least {min_required} hours, "
                f"but only {len(cost_df)} available"
        )
            sys.exit(1)
        else:
            logger.info(
            f"✅ Data sufficient: {len(cost_df)} >= {min_required}"
        )
# ------------------------------------------------------------------
        

        # Step 2: Create dataset
        logger.info("\n[2/6] Creating dataset...")
        dataset = CostTimeSeriesDataset(
            cost_df,
            seq_length=SEQUENCE_LENGTH,
            forecast_length=FORECAST_LENGTH,
            logger=logger
        )

        # Step 3: Split data
        logger.info("\n[3/6] Splitting data...")
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size]
        )

        logger.info(f"  Train: {len(train_dataset):,} sequences")
        logger.info(f"  Val: {len(val_dataset):,} sequences")

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Step 4: Initialize model
        logger.info("\n[4/6] Initializing model...")
        model = CostForecaster(
            input_dim=6,  # cpu, memory, storage, network, duration, cost
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
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")

        # Step 5: Train model
        logger.info("\n[5/6] Training model...")
        train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            dataset,
            logger,
            epochs=EPOCHS,
            lr=LEARNING_RATE
        )

        # Save scaler
        logger.info("\nSaving scaler...")
        scaler_path = MODELS_DIR / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(dataset.scaler, f)
        logger.info(f"[OK] Scaler saved to {scaler_path}")

        # Step 6: Calculate accuracy
        logger.info("\n[6/6] Calculating accuracy metrics...")
        y_true, y_pred, accuracy_pct = calculate_accuracy(model, val_loader, dataset, logger)

        # Generate plots
        plot_training_curves(train_losses, val_losses, logger)
        plot_accuracy_scatter(y_true, y_pred, accuracy_pct, logger)

        # Final summary
        logger.info("\n" + "="*70)
        logger.info("[SUCCESS] TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"Model saved to: {MODELS_DIR / 'transformer_model.pth'}")
        logger.info(f"Scaler saved to: {scaler_path}")
        logger.info(f"Training curve: {MODELS_DIR / 'training_curve.png'}")
        logger.info(f"Accuracy plot: {MODELS_DIR / 'accuracy_scatter.png'}")
        logger.info(f"Model Accuracy: {accuracy_pct:.2f}%")
        logger.info("\nYou can now use the model for forecasting!")
        logger.info("Restart your Flask server to load the new model.")
        logger.info("="*70)

        return 0

    except FileNotFoundError as e:
        logger.error(f"\n[ERROR] File not found: {str(e)}")
        logger.error("Please run preprocessing first: python scripts/preprocess_data.py")
        return 1
    except Exception as e:
        logger.exception(f"\n[ERROR] Unexpected error: {str(e)}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)