import os
from pathlib import Path


class Config:
    """
    Configuration settings for the application
    All paths and constants in one place
    """

    # ==========================================
    # Base Directories
    # ==========================================
    BASE_DIR = Path(__file__).parent.parent  # backend/
    DATA_DIR = BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    MODELS_DIR = BASE_DIR / 'trained_models'

    # ==========================================
    # Flask Settings
    # ==========================================
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-prod')
    DEBUG = os.environ.get('FLASK_ENV', 'development') == 'development'
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))

    # ==========================================
    # Model Paths
    # ==========================================
    TRANSFORMER_MODEL_PATH = MODELS_DIR / 'transformer_model.pth'
    RL_AGENT_PATH = MODELS_DIR / 'rl_agent.pth'
    SCALER_PATH = MODELS_DIR / 'scaler.pkl'

    # ==========================================
    # Data Files
    # ==========================================
    # Raw data
    GOOGLE_TRACE_FILE = RAW_DATA_DIR / 'workloads' / 'instance_usage-000000000000.json'
    AWS_EC2_PRICING_FILE = RAW_DATA_DIR / 'pricing' / 'aws_ec2_pricing.json'
    AZURE_VM_PRICING_FILE = RAW_DATA_DIR / 'pricing' / 'azure_vm_pricing.json'

    # Processed data
    GOOGLE_SAMPLE_FILE = PROCESSED_DATA_DIR / 'google_trace_sample.csv'
    AWS_SAMPLE_FILE = PROCESSED_DATA_DIR / 'aws_ec2_sample.csv'
    COST_TIMESERIES_FILE = PROCESSED_DATA_DIR / 'cost_timeseries.csv'
    NORMALIZED_DATA_FILE = PROCESSED_DATA_DIR / 'normalized_data.pkl'
    
    # ==========================================
    # Model Hyperparameters
    # ==========================================
    # Transformer
    TRANSFORMER_INPUT_DIM = 4  # cpu, memory, duration, cost
    TRANSFORMER_D_MODEL = 64
    TRANSFORMER_NHEAD = 4
    TRANSFORMER_NUM_ENCODER_LAYERS = 3
    TRANSFORMER_NUM_DECODER_LAYERS = 3
    TRANSFORMER_DIM_FEEDFORWARD = 256
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_OUTPUT_DIM = 1  # predict cost only

    # RL Agent
    RL_STATE_DIM = 6  # cpu, memory, network, cost, budget, provider
    RL_ACTION_DIM = 4  # scale_up, scale_down, migrate, no_change
    RL_HIDDEN_DIM = 128
    RL_LEARNING_RATE = 0.001
    RL_GAMMA = 0.99  # discount factor
    RL_EPSILON_START = 1.0
    RL_EPSILON_END = 0.01
    RL_EPSILON_DECAY = 0.995

    # ==========================================
    # Training Parameters
    # ==========================================
    SEQUENCE_LENGTH = 168  # hours of historical data
    FORECAST_LENGTH = 24  # hours to predict
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001

    # ==========================================
    # API Settings
    # ==========================================
    MAX_FORECAST_HOURS = 24  # max hours to forecast
    MIN_FORECAST_HOURS = 1  # min hours to forecast
    DEFAULT_FORECAST_HOURS = 24

    VALID_CLOUD_PROVIDERS = ['aws', 'azure']
    VALID_SERVICE_TYPES = ['vm', 'storage', 'network', 'kubernetes']

    # Rate limiting
    MAX_REQUESTS_PER_MINUTE = 100

    # ==========================================
    # Cloud Provider Settings
    # ==========================================
    # Average pricing (used for calculations)
    AVG_AWS_VM_PRICE = 0.10  # USD per hour
    AVG_AZURE_VM_PRICE = 0.12  # USD per hour
    AVG_AWS_STORAGE_PRICE = 0.023  # USD per GB per month
    AVG_AZURE_STORAGE_PRICE = 0.026  # USD per GB per month

    # ==========================================
    # Logging
    # ==========================================
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = BASE_DIR / 'app.log'

    # ==========================================
    # Helper Methods
    # ==========================================

    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.RAW_DATA_DIR / 'workloads',
            cls.RAW_DATA_DIR / 'pricing',
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        print("✓ All directories created/verified")

    @classmethod
    def check_files_exist(cls):
        """Check which required files exist"""
        files_to_check = {
            'Google Trace Sample': cls.GOOGLE_SAMPLE_FILE,
            'AWS EC2 Sample': cls.AWS_SAMPLE_FILE,
            'Cost Timeseries': cls.COST_TIMESERIES_FILE,
            'Normalized Data': cls.NORMALIZED_DATA_FILE,
            'Transformer Model': cls.TRANSFORMER_MODEL_PATH,
            'Scaler': cls.SCALER_PATH,
        }

        status = {}
        all_exist = True

        print("\nFile Status Check:")
        print("=" * 60)

        for name, path in files_to_check.items():
            exists = path.exists()
            status[name] = exists
            all_exist = all_exist and exists

            symbol = "✓" if exists else "✗"
            print(f"{symbol} {name}: {path}")

        print("=" * 60)

        if all_exist:
            print("✓ All required files present!")
        else:
            print("⚠ Some files are missing. Run preprocessing/training scripts.")

        return status

    @classmethod
    def get_device(cls):
        """Get the computing device (CPU or CUDA)"""
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        return device

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "=" * 60)
        print("CONFIGURATION")
        print("=" * 60)
        print(f"Environment: {'Development' if cls.DEBUG else 'Production'}")
        print(f"Host: {cls.HOST}:{cls.PORT}")
        print(f"Base Directory: {cls.BASE_DIR}")
        print(f"Models Directory: {cls.MODELS_DIR}")
        print(f"Data Directory: {cls.DATA_DIR}")
        print(f"\nTransformer Config:")
        print(f"  - Input Dim: {cls.TRANSFORMER_INPUT_DIM}")
        print(f"  - Model Dim: {cls.TRANSFORMER_D_MODEL}")
        print(f"  - Attention Heads: {cls.TRANSFORMER_NHEAD}")
        print(f"  - Encoder Layers: {cls.TRANSFORMER_NUM_ENCODER_LAYERS}")
        print(f"  - Decoder Layers: {cls.TRANSFORMER_NUM_DECODER_LAYERS}")
        print(f"\nTraining Config:")
        print(f"  - Sequence Length: {cls.SEQUENCE_LENGTH} hours")
        print(f"  - Forecast Length: {cls.FORECAST_LENGTH} hours")
        print(f"  - Batch Size: {cls.BATCH_SIZE}")
        print(f"  - Epochs: {cls.EPOCHS}")
        print(f"  - Learning Rate: {cls.LEARNING_RATE}")
        print(f"\nAPI Config:")
        print(f"  - Max Forecast Days: {cls.MAX_FORECAST_HOURS}")
        print(f"  - Valid Providers: {', '.join(cls.VALID_CLOUD_PROVIDERS)}")
        print(f"  - Valid Services: {', '.join(cls.VALID_SERVICE_TYPES)}")
        print("=" * 60 + "\n")


# Initialize directories when config is imported
Config.ensure_directories()

# Development/Testing utilities
if __name__ == '__main__':
    """
    Run this file directly to check configuration
    Usage: python utils/config.py
    """
    Config.print_config()
    Config.check_files_exist()