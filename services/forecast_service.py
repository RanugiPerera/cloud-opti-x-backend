import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import pickle

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from scripts.models.transformer import CostForecaster
from utils.config import Config


class ForecastService:
    """
    Service for cost forecasting using Transformer model
    Predicts future cloud costs based on historical data
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_loaded = False

        # load model and scaler
        self._load_model()
        self._load_scaler()

    def _load_model(self):
        """Load trained Transformer model"""
        try:
            model_path = Config.TRANSFORMER_MODEL_PATH

            if not Path(model_path).exists():
                print(f"⚠ Warning: Model not found at {model_path}")
                print("Predictions will use fallback method")
                return

            # initialize model with 6 input features
            self.model = CostForecaster(
                input_dim=6,  # CHANGED FROM 4 TO 6
                d_model=64,
                nhead=4,
                num_encoder_layers=3,
                num_decoder_layers=3,
                dim_feedforward=256,
                dropout=0.1,
                output_dim=1,
                device=self.device
            )

            # load weights
            self.model.load(model_path)
            self.model.model.eval()

            print(f"✓ Forecast model loaded from {model_path}")
            self.is_loaded = True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.is_loaded = False

    def _load_scaler(self):
        """Load the data scaler"""
        try:
            scaler_path = Config.SCALER_PATH

            if not Path(scaler_path).exists():
                print(f"⚠ Warning: Scaler not found at {scaler_path}")
                print("Predictions will use raw values")
                return

            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

            print(f"✓ Scaler loaded from {scaler_path}")

        except Exception as e:
            print(f"Error loading scaler: {str(e)}")
            self.scaler = None

    def _prepare_input_data(self, cloud_provider='aws', service_type='vm', lookback_hours=24):
        """
        Prepare input data for prediction
        Uses actual historical data if available, otherwise creates realistic synthetic data
        """

        # try to load historical cost data
        try:
            cost_df = pd.read_csv(Config.PROCESSED_DATA_DIR / 'cost_timeseries.csv')

            # filter by provider and service
            filtered = cost_df[
                (cost_df['cloud_provider'] == cloud_provider) &
                (cost_df['service_type'] == service_type)
            ].sort_values('hour')

            if len(filtered) >= lookback_hours:
                # use actual historical data with 6 features
                recent_data = filtered.tail(lookback_hours)
                features = recent_data[['cpu_usage', 'memory_usage', 'storage_gb', 'network_gb', 'duration_hours', 'cost']].values

                print(f"Using historical data: {len(features)} records")
                print(f"Cost range: ${features[:, -1].min():.4f} - ${features[:, -1].max():.4f}")

                return features

        except Exception as e:
            print(f"Could not load historical data: {str(e)}")

        # fallback: create synthetic data
        print("Using synthetic data for prediction")
        return self._create_synthetic_input(lookback_hours, cloud_provider, service_type)

    def _create_synthetic_input(self, lookback_hours=24, cloud_provider='aws', service_type='vm'):
        """
        Create realistic synthetic input data with 6 features
        """
        # generate realistic-looking patterns
        time = np.arange(lookback_hours)

        # cpu usage with daily pattern + noise
        cpu = 0.4 + 0.2 * np.sin(2 * np.pi * time / 24) + np.random.normal(0, 0.05, lookback_hours)
        cpu = np.clip(cpu, 0.1, 0.9)

        # memory usage (correlated with cpu)
        memory = cpu * 0.7 + np.random.normal(0, 0.03, lookback_hours)
        memory = np.clip(memory, 0.1, 0.9)

        # storage (slowly increasing)
        storage = 100 + np.cumsum(np.random.uniform(-2, 3, lookback_hours))
        storage = np.clip(storage, 50, 300)

        # network usage (varies with time)
        network = 5 + 3 * np.sin(2 * np.pi * time / 12) + np.random.normal(0, 1, lookback_hours)
        network = np.clip(network, 1, 15)

        # duration (mostly 1 hour with some variation)
        duration = np.ones(lookback_hours) + np.random.normal(0, 0.1, lookback_hours)
        duration = np.clip(duration, 0.5, 2.0)

        # calculate cost based on resources
        if cloud_provider == 'aws':
            cost = (cpu * 0.05 + memory * 0.02 +
                   storage * 0.1 / 730 + network * 0.09) * duration
        else:
            cost = (cpu * 0.055 + memory * 0.022 +
                   storage * 0.12 / 730 + network * 0.087) * duration

        # add service type multiplier
        if service_type == 'kubernetes':
            cost *= 1.3
        elif service_type == 'storage':
            cost *= 0.8

        # stack all features
        features = np.column_stack([cpu, memory, storage, network, duration, cost])

        return features

    def predict_costs(self, cloud_provider='aws', service_type='vm', forecast_hours=24):
        """
        Predict future costs

        Args:
            cloud_provider: 'aws' or 'azure'
            service_type: 'vm', 'storage', 'network', or 'kubernetes'
            forecast_hours: number of hours to forecast (1-24)

        Returns:
            dict with predictions and metadata
        """

        if not self.is_loaded:
            return {
                'error': 'Model not loaded. Please train the model first.',
                'predictions': []
            }

        try:
            print(f"\n{'=' * 60}")
            print(f"Making prediction for {cloud_provider} {service_type}")
            print(f"{'=' * 60}")

            # prepare input data (last 24 hours) with 6 features
            input_data = self._prepare_input_data(
                cloud_provider=cloud_provider,
                service_type=service_type,
                lookback_hours=24
            )

            print(f"\nInput data shape: {input_data.shape}")
            print(f"Input cost mean (original): ${input_data[:, -1].mean():.4f}")

            # normalize input data using the scaler
            if self.scaler is not None:
                print("Normalizing input data...")
                input_normalized = self.scaler.transform(input_data)
                print(f"Input cost mean (normalized): {input_normalized[:, -1].mean():.4f}")
            else:
                print("⚠ No scaler available, using raw data")
                input_normalized = input_data

            # reshape for model: (1, seq_len, features) - now 6 features
            input_tensor = input_normalized.reshape(1, -1, 6)

            print(f"Input tensor shape: {input_tensor.shape}")

            # make prediction
            print("Running model inference...")
            predictions = self.model.predict(
                input_tensor,
                forecast_hours
            )

            print(f"Raw predictions shape: {predictions.shape}")
            print(f"Raw predictions (normalized): {predictions[0, :, 0]}")

            # extract predictions (these are normalized)
            predictions_normalized = predictions[0, :, 0]

            # denormalize predictions back to original scale
            if self.scaler is not None:
                print("Denormalizing predictions...")

                # for cost (last column), use the min/max from the scaler
                cost_min = self.scaler.data_min_[-1]
                cost_range = self.scaler.data_range_[-1]

                # denormalize: original = normalized * range + min
                predictions_denorm = predictions_normalized * cost_range + cost_min

                print(f"Scaler info:")
                print(f"  Cost min: ${cost_min:.4f}")
                print(f"  Cost range: ${cost_range:.4f}")
                print(f"Predictions (denormalized): {predictions_denorm}")
            else:
                print("⚠ No scaler, using predictions as-is")
                predictions_denorm = predictions_normalized

            # ensure positive costs
            predictions_denorm = np.maximum(predictions_denorm, 0.0001)

            print(f"Final predictions: {predictions_denorm}")
            print(f"  Mean: ${predictions_denorm.mean():.4f}")
            print(f"  Range: ${predictions_denorm.min():.4f} - ${predictions_denorm.max():.4f}")

            # create time labels
            time_labels = [f"+{i + 1}h" for i in range(forecast_hours)]

            # calculate confidence intervals (±15%)
            lower_bound = predictions_denorm * 0.85
            upper_bound = predictions_denorm * 1.15

            result = {
                'status': 'success',
                'cloud_provider': cloud_provider,
                'service_type': service_type,
                'forecast_horizon': forecast_hours,
                'predictions': predictions_denorm.tolist(),
                'time_labels': time_labels,
                'confidence_interval': {
                    'lower': lower_bound.tolist(),
                    'upper': upper_bound.tolist()
                },
                'average_predicted_cost': float(np.mean(predictions_denorm)),
                'total_predicted_cost': float(np.sum(predictions_denorm)),
                'min_cost': float(np.min(predictions_denorm)),
                'max_cost': float(np.max(predictions_denorm)),
                'unit': 'USD per hour',
                'prediction_method': 'transformer (ML)',
                'note': 'Predictions using trained Transformer model with 6 features'
            }

            print(f"\n✓ Prediction complete!")
            print(f"  Average: ${result['average_predicted_cost']:.4f}/hour")
            print(f"  Total: ${result['total_predicted_cost']:.4f}")
            print(f"{'=' * 60}\n")

            return result

        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()

            return {
                'error': f'Prediction failed: {str(e)}',
                'predictions': []
            }

    def get_summary_stats(self):
        """
        Get summary statistics about the model
        """

        if not self.is_loaded:
            return {'error': 'Model not loaded'}

        try:
            # load historical data
            cost_df = pd.read_csv(Config.PROCESSED_DATA_DIR / 'cost_timeseries.csv')

            stats = {
                'model_loaded': True,
                'device': self.device,
                'features': 6,  # now using 6 features
                'historical_data_points': len(cost_df),
                'providers': cost_df['cloud_provider'].unique().tolist(),
                'services': cost_df['service_type'].unique().tolist(),
                'avg_cost_per_hour': {
                    provider: float(cost_df[cost_df['cloud_provider'] == provider]['cost'].mean())
                    for provider in cost_df['cloud_provider'].unique()
                },
                'total_cost': float(cost_df['cost'].sum()),
                'date_range': {
                    'start': int(cost_df['hour'].min()),
                    'end': int(cost_df['hour'].max())
                }
            }

            return stats

        except Exception as e:
            return {'error': f'Failed to get stats: {str(e)}'}


# create singleton instance
forecast_service = ForecastService()