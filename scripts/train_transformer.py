"""
Transformer Model Training - Cloud Cost Forecasting
FIXED: Replaced broken MAPE with proper metrics (MAE, RMSE, R²)
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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
SEQUENCE_LENGTH = 168  # use past 168 hours to predict
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
    """
    Calculate final accuracy metrics using PROPER metrics (not broken MAPE)
    """
    
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

    # ============================================
    # PROPER METRICS (NOT BROKEN MAPE!)
    # ============================================

    # 1. R² Score (Variance Explained) - Primary metric
    r2 = r2_score(y_true_dollars, y_pred_dollars)

    # 2. Mean Absolute Error - Easy to interpret
    mae = mean_absolute_error(y_true_dollars, y_pred_dollars)

    # 3. Root Mean Squared Error - Penalizes large errors
    rmse = np.sqrt(mean_squared_error(y_true_dollars, y_pred_dollars))

    # 4. Median Absolute Error - Robust to outliers
    median_ae = np.median(np.abs(y_true_dollars - y_pred_dollars))

    # 5. Percentage-based metrics (using MAE)
    avg_cost = y_true_dollars.mean()
    mae_percentage = (mae / avg_cost) * 100 if avg_cost > 0 else 0

    # 6. OPTIONAL: Filtered MAPE (only for costs > $1.00 to avoid division issues)
    mask = y_true_dollars > 1.00
    if np.sum(mask) > 10:  # Only if we have enough high-value samples
        mape_filtered = np.mean(np.abs((y_true_dollars[mask] - y_pred_dollars[mask]) 
                                        / y_true_dollars[mask])) * 100
        coverage_pct = 100 * np.sum(mask) / len(y_true_dollars)
    else:
        mape_filtered = None
        coverage_pct = 0

    # ============================================
    # PRINT COMPREHENSIVE REPORT
    # ============================================

    logger.info("\n" + "="*70)
    logger.info("MODEL PERFORMANCE METRICS")
    logger.info("="*70)

    logger.info(f"\n PRIMARY METRICS:")
    logger.info(f"  R² Score (Variance Explained):    {r2:.4f} ({r2*100:.2f}%)")
    logger.info(f"  Mean Absolute Error (MAE):        ${mae:.4f}")
    logger.info(f"  Root Mean Squared Error (RMSE):   ${rmse:.4f}")
    logger.info(f"  Median Absolute Error:            ${median_ae:.4f}")

    logger.info(f"\n PERCENTAGE METRICS:")
    logger.info(f"  MAE as % of average cost:         {mae_percentage:.2f}%")
    
    if mape_filtered is not None and coverage_pct > 50:
        logger.info(f"  MAPE (costs > $1.00 only):        {mape_filtered:.2f}%")
        logger.info(f"  Coverage:                         {coverage_pct:.1f}% of data")
        logger.info(f"  Accuracy (from filtered MAPE):    {100 - mape_filtered:.2f}%")
    else:
        logger.info(f"  MAPE: Not computed (insufficient high-value samples)")

    logger.info(f"\n COST STATISTICS:")
    logger.info(f"  Mean Predicted Cost:              ${y_pred_dollars.mean():.4f}")
    logger.info(f"  Mean Actual Cost:                 ${y_true_dollars.mean():.4f}")
    logger.info(f"  Prediction Range:                 ${y_pred_dollars.min():.4f} - ${y_pred_dollars.max():.4f}")
    logger.info(f"  Actual Range:                     ${y_true_dollars.min():.4f} - ${y_true_dollars.max():.4f}")


    # ============================================
    # DATA DISTRIBUTION ANALYSIS
    # ============================================

    logger.info("="*70)
    logger.info("DATA DISTRIBUTION ANALYSIS")
    logger.info("="*70)

    bins = [0, 0.10, 0.50, 1.00, 5.00, 10.00, float('inf')]
    labels = ['$0-0.10', '$0.10-0.50', '$0.50-1.00', '$1.00-5.00', '$5.00-10.00', '>$10']

    logger.info("\nActual cost distribution:")
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        count = np.sum((y_true_dollars >= low) & (y_true_dollars < high))
        pct = 100 * count / len(y_true_dollars)
        logger.info(f"  {labels[i]:>15}: {count:>6} samples ({pct:>5.1f}%)")

    near_zero = np.sum(y_true_dollars < 0.50)
    near_zero_pct = 100 * near_zero / len(y_true_dollars)
    logger.info(f"\n  Values < $0.50: {near_zero} ({near_zero_pct:.1f}%)")
    logger.info("These low values would break traditional MAPE calculation")
    logger.info("That's why we use MAE and R² as primary metrics instead")

    logger.info("="*70 + "\n")

    return y_true_dollars, y_pred_dollars, r2, mae, mae_percentage


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
    logger.info(f"✓ Training curve saved to: {plot_path}")
    plt.close()


def plot_accuracy_scatter(y_true, y_pred, r2, mae, logger):
    """Generate and save accuracy scatter plot with proper metrics"""
    
    plt.figure(figsize=(10, 10))
    
    # Main scatter plot
    plt.scatter(y_true, y_pred, alpha=0.3, color='teal', s=20)
    
    # Add identity line (perfect predictions)
    max_val = max(y_true.max(), y_pred.max())
    min_val = min(y_true.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], 
             color='red', lw=2, linestyle='--', label='Perfect Prediction')
    
    # Add ±20% error bands
    plt.plot([min_val, max_val], [min_val*0.8, max_val*0.8], 
             color='orange', lw=1, linestyle=':', alpha=0.5, label='±20% Error')
    plt.plot([min_val, max_val], [min_val*1.2, max_val*1.2], 
             color='orange', lw=1, linestyle=':', alpha=0.5)
    
    plt.xlabel('Actual Cost ($)', fontsize=12)
    plt.ylabel('Predicted Cost ($)', fontsize=12)
    plt.title(f'Actual vs Predicted Cost\nR²: {r2:.4f} | MAE: ${mae:.4f}', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = MODELS_DIR / 'accuracy_scatter.png'
    plt.savefig(plot_path, dpi=150)
    logger.info(f"✓ Accuracy scatter plot saved to: {plot_path}")
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
        logger.info(f"✓ Loaded {len(cost_df):,} records")

        # Data sufficiency check
        min_required = SEQUENCE_LENGTH + FORECAST_LENGTH

        logger.info("\nData requirements check:")
        logger.info(f"  Sequence length: {SEQUENCE_LENGTH}")
        logger.info(f"  Forecast length: {FORECAST_LENGTH}")
        logger.info(f"  Total required hours: {min_required}")
        logger.info(f"  Available data hours: {len(cost_df)}")

        if len(cost_df) < min_required:
            logger.error(
                f"Insufficient data: need at least {min_required} hours, "
                f"but only {len(cost_df)} available"
            )
            sys.exit(1)
        else:
            logger.info(f"Data sufficient: {len(cost_df)} >= {min_required}")

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
        logger.info(f"✓ Scaler saved to {scaler_path}")

        # Step 6: Calculate accuracy with PROPER metrics
        logger.info("\n[6/6] Calculating accuracy metrics...")
        y_true, y_pred, r2, mae, mae_pct= calculate_accuracy(
            model, val_loader, dataset, logger
        )

        # Generate plots
        plot_training_curves(train_losses, val_losses, logger)
        plot_accuracy_scatter(y_true, y_pred, r2, mae, logger)

        # Final summary
        logger.info("\n" + "="*70)
        logger.info(" TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"\n Saved Files:")
        logger.info(f"  Model:         {MODELS_DIR / 'transformer_model.pth'}")
        logger.info(f"  Scaler:        {scaler_path}")
        logger.info(f"  Training plot: {MODELS_DIR / 'training_curve.png'}")
        logger.info(f"  Accuracy plot: {MODELS_DIR / 'accuracy_scatter.png'}")
        
        logger.info(f"\n Final Metrics:")
        logger.info(f"  R² Score:      {r2:.4f} ({r2*100:.1f}%)")
        logger.info(f"  MAE:           ${mae:.4f} ({mae_pct:.1f}% of avg)")

        return 0

    except FileNotFoundError as e:
        logger.error(f"\n File not found: {str(e)}")
        logger.error("Please run preprocessing first: python scripts/preprocess_data.py")
        return 1
    except Exception as e:
        logger.exception(f"\n Unexpected error: {str(e)}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)