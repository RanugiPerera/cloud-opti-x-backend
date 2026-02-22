"""
Enhanced Cloud Cost Data Preprocessing Pipeline
Processes Google Cluster Trace and Cloud Pricing data for ML model training
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Centralized configuration for preprocessing pipeline"""
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    RAW_DIR: Path = BASE_DIR / 'data' / 'raw'
    PROCESSED_DIR: Path = BASE_DIR / 'data' / 'processed'
    MODELS_DIR: Path = BASE_DIR / 'trained_models'
    LOGS_DIR: Path = BASE_DIR / 'logs'
    
    # Sampling parameters
    GOOGLE_SAMPLE_SIZE: int = 500_000
    AWS_SAMPLE_SIZE: int = 50_000
    PROGRESS_INTERVAL: int = 1_000_000
    RANDOM_SEED: Optional[int] = None  # Set to int for reproducibility
    
    # Pricing (AWS)
    AWS_CPU_PRICE: float = 0.05
    AWS_MEMORY_PRICE: float = 0.02
    AWS_STORAGE_PRICE: float = 0.1 / 730
    AWS_NETWORK_PRICE: float = 0.09
    
    # Pricing (Azure)
    AZURE_CPU_PRICE: float = 0.055
    AZURE_MEMORY_PRICE: float = 0.022
    AZURE_STORAGE_PRICE: float = 0.12 / 730
    AZURE_NETWORK_PRICE: float = 0.087
    
    # Resource simulation ranges
    STORAGE_MIN_GB: float = 10
    STORAGE_MAX_GB: float = 500
    NETWORK_MIN_RATE: float = 0.5
    NETWORK_MAX_RATE: float = 5.0
    
    # Service type distribution
    SERVICE_TYPES: list = None
    SERVICE_PROBABILITIES: list = None
    
    # Overhead multipliers
    KUBERNETES_OVERHEAD: float = 1.3
    STORAGE_DISCOUNT: float = 0.8
    
    def __post_init__(self):
        """Initialize derived attributes"""
        if self.SERVICE_TYPES is None:
            self.SERVICE_TYPES = ['vm', 'storage', 'network', 'kubernetes']
        if self.SERVICE_PROBABILITIES is None:
            self.SERVICE_PROBABILITIES = [0.5, 0.2, 0.2, 0.1]
        
        # Create directories
        for directory in [self.PROCESSED_DIR, self.MODELS_DIR, self.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Set random seed if specified
        if self.RANDOM_SEED is not None:
            np.random.seed(self.RANDOM_SEED)
            logging.info(f"Random seed set to: {self.RANDOM_SEED}")


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config: Config) -> logging.Logger:
    """Configure logging with both file and console handlers"""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = config.LOGS_DIR / f'preprocessing_{timestamp}.log'
    
    # Create logger
    logger = logging.getLogger('preprocessing')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler
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
# UTILITY FUNCTIONS
# ============================================================================

def validate_file_exists(filepath: Path, logger: logging.Logger) -> bool:
    """Validate that input file exists"""
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return False
    
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    logger.info(f"Found file: {filepath.name} ({file_size_mb:.1f} MB)")
    return True


def save_dataframe(df: pd.DataFrame, filepath: Path, logger: logging.Logger) -> None:
    """Save DataFrame with validation and logging"""
    try:
        df.to_csv(filepath, index=False)
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"[OK] Saved: {filepath.name} ({len(df):,} rows, {file_size_mb:.1f} MB)")
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {str(e)}")
        raise


def validate_dataframe(df: pd.DataFrame, required_cols: list, logger: logging.Logger) -> bool:
    """Validate DataFrame has required columns and data"""
    if df.empty:
        logger.error("DataFrame is empty")
        return False
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    # Check for null values
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        logger.warning(f"Null values found:\n{null_counts[null_counts > 0]}")
    
    return True


# ============================================================================
# STEP 1: SAMPLE GOOGLE CLUSTER TRACE
# ============================================================================

def sample_google_trace(
    input_file: Path,
    output_file: Path,
    config: Config,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Sample Google cluster usage data with improved error handling and efficiency
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to save sampled CSV
        config: Configuration object
        logger: Logger instance
    
    Returns:
        Sampled DataFrame
    """
    logger.info(f"\n{'='*60}")
    logger.info("[1/6] Sampling Google Cluster Trace")
    logger.info(f"{'='*60}")
    logger.info(f"Target sample size: {config.GOOGLE_SAMPLE_SIZE:,}")
    
    # Validate input
    if not validate_file_exists(input_file, logger):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Count total lines
    logger.info("Counting total lines...")
    total_lines = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for _ in f:
                total_lines += 1
    except Exception as e:
        logger.error(f"Error counting lines: {str(e)}")
        raise
    
    logger.info(f"Total lines: {total_lines:,}")
    
    # Calculate sampling rate
    sample_rate = min(1.0, config.GOOGLE_SAMPLE_SIZE / total_lines) if total_lines > 0 else 0
    logger.info(f"Sampling rate: {sample_rate:.2%}")
    
    # Random sampling with progress tracking
    data = []
    parse_errors = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Progress reporting
                if (i + 1) % config.PROGRESS_INTERVAL == 0:
                    logger.info(
                        f"  Progress: {i + 1:,} lines processed, "
                        f"{len(data):,} samples collected, "
                        f"{parse_errors} errors"
                    )
                
                # Random sampling
                if np.random.random() < sample_rate:
                    try:
                        row = json.loads(line.strip())
                        data.append(row)
                    except json.JSONDecodeError:
                        parse_errors += 1
                        if parse_errors < 10:  # Log first few errors
                            logger.debug(f"JSON parse error at line {i+1}")
                        continue
                
                # Early stopping if we have enough samples
                if len(data) >= config.GOOGLE_SAMPLE_SIZE:
                    logger.info(f"Reached target sample size at line {i+1:,}")
                    break
    
    except Exception as e:
        logger.error(f"Error during sampling: {str(e)}")
        raise
    
    logger.info(f"Collected {len(data):,} samples ({parse_errors} parse errors)")
    
    if not data:
        raise ValueError("No valid data collected from input file")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    logger.info(f"DataFrame created with {len(df):,} rows, {len(df.columns)} columns")
    
    # Select relevant columns
    relevant_cols = [
        'start_time', 'end_time', 'collection_id', 'instance_index',
        'machine_id', 'average_usage', 'maximum_usage'
    ]
    
    existing_cols = [col for col in relevant_cols if col in df.columns]
    logger.info(f"Keeping columns: {existing_cols}")
    df = df[existing_cols]
    
    # Extract nested CPU and memory usage
    if 'average_usage' in df.columns:
        if isinstance(df['average_usage'].iloc[0], dict):
            logger.info("Extracting nested CPU and memory usage...")
            
            df['cpu_usage'] = df['average_usage'].apply(
                lambda x: x.get('cpus', 0) if isinstance(x, dict) else 0
            )
            df['memory_usage'] = df['average_usage'].apply(
                lambda x: x.get('memory', 0) if isinstance(x, dict) else 0
            )
            
            df = df.drop('average_usage', axis=1)
            
            # Log statistics
            logger.info(f"CPU usage - Mean: {df['cpu_usage'].mean():.4f}, "
                       f"Min: {df['cpu_usage'].min():.4f}, "
                       f"Max: {df['cpu_usage'].max():.4f}")
            logger.info(f"Memory usage - Mean: {df['memory_usage'].mean():.4f}, "
                       f"Min: {df['memory_usage'].min():.4f}, "
                       f"Max: {df['memory_usage'].max():.4f}")
    
    # Validate output
    required_cols = ['start_time', 'end_time', 'cpu_usage', 'memory_usage']
    if not validate_dataframe(df, required_cols, logger):
        raise ValueError("DataFrame validation failed")
    
    # Save
    save_dataframe(df, output_file, logger)
    
    return df


# ============================================================================
# STEP 2: SAMPLE AWS PRICING
# ============================================================================

def sample_aws_pricing(
    input_file: Path,
    output_file: Path,
    config: Config,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Sample AWS EC2 pricing data with improved structure handling
    """
    logger.info(f"\n{'='*60}")
    logger.info("[2/6] Sampling AWS EC2 Pricing")
    logger.info(f"{'='*60}")
    logger.info(f"Target sample size: {config.AWS_SAMPLE_SIZE:,}")
    
    # Validate input
    if not validate_file_exists(input_file, logger):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    try:
        # Load JSON
        logger.info("Loading AWS pricing file...")
        with open(input_file, 'r', encoding='utf-8') as f:
            aws_data = json.load(f)
        
        logger.info(f"[OK] JSON loaded successfully")
        logger.info(f"  Top-level keys: {list(aws_data.keys())}")
        
        # Extract products
        if 'products' not in aws_data:
            logger.error("'products' key not found in JSON")
            return pd.DataFrame()
        
        products = aws_data['products']
        logger.info(f"  Found {len(products):,} products")
        
        # Convert to list of records
        data = []
        for sku, product_info in products.items():
            record = {
                'sku': sku,
                'productFamily': product_info.get('productFamily', ''),
            }
            
            # Extract attributes
            if 'attributes' in product_info:
                attrs = product_info['attributes']
                record.update({
                    'serviceCode': attrs.get('servicecode', ''),
                    'location': attrs.get('location', ''),
                    'locationType': attrs.get('locationType', ''),
                    'instanceType': attrs.get('instanceType', ''),
                    'vcpu': attrs.get('vcpu', ''),
                    'memory': attrs.get('memory', ''),
                    'storage': attrs.get('storage', ''),
                    'networkPerformance': attrs.get('networkPerformance', ''),
                    'physicalProcessor': attrs.get('physicalProcessor', ''),
                    'clockSpeed': attrs.get('clockSpeed', ''),
                    'operatingSystem': attrs.get('operatingSystem', ''),
                    'licenseModel': attrs.get('licenseModel', ''),
                    'tenancy': attrs.get('tenancy', ''),
                    'usagetype': attrs.get('usagetype', ''),
                })
            
            data.append(record)
        
        logger.info(f"  Converted to {len(data):,} records")
        
        # Sample if needed
        if len(data) > config.AWS_SAMPLE_SIZE:
            logger.info(f"  Sampling {config.AWS_SAMPLE_SIZE:,} random records...")
            indices = np.random.choice(len(data), config.AWS_SAMPLE_SIZE, replace=False)
            data = [data[i] for i in indices]
        
        logger.info(f"[OK] Final sample size: {len(data):,} rows")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        logger.info(f"DataFrame created:")
        logger.info(f"  Rows: {len(df):,}")
        logger.info(f"  Columns: {len(df.columns)}")
        
        # Log value distribution for key columns
        if 'instanceType' in df.columns:
            logger.info(f"  Unique instance types: {df['instanceType'].nunique()}")
        if 'location' in df.columns:
            logger.info(f"  Unique locations: {df['location'].nunique()}")
        
        # Save
        save_dataframe(df, output_file, logger)
        
        return df
    
    except FileNotFoundError:
        logger.error(f"File not found: {input_file}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading AWS pricing: {str(e)}")
        raise


# ============================================================================
# STEP 3: LOAD SMALL PRICING FILES
# ============================================================================

def load_small_pricing_files(
    config: Config,
    logger: logging.Logger
) -> Dict[str, dict]:
    """
    Load Azure and other AWS pricing files with error handling
    """
    logger.info(f"\n{'='*60}")
    logger.info("[3/6] Loading Additional Pricing Files")
    logger.info(f"{'='*60}")
    
    pricing_data = {}
    
    pricing_files = {
        'azure_vm': config.RAW_DIR / 'pricing' / 'azure_vm_pricing.json',
        'azure_storage': config.RAW_DIR / 'pricing' / 'azure_storage_pricing.json',
        'azure_vnet': config.RAW_DIR / 'pricing' / 'azure_vnet_pricing.json',
        'azure_aks': config.RAW_DIR / 'pricing' / 'azure_aks_pricing.json',
        'aws_s3': config.RAW_DIR / 'pricing' / 'aws_s3_pricing.json',
        'aws_vpc': config.RAW_DIR / 'pricing' / 'aws_vpc_pricing.json',
        'aws_eks': config.RAW_DIR / 'pricing' / 'aws_eks_pricing.json',
    }
    
    loaded_count = 0
    for name, filepath in pricing_files.items():
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    pricing_data[name] = json.load(f)
                logger.info(f"  [OK] Loaded: {name}")
                loaded_count += 1
            except Exception as e:
                logger.warning(f"  [FAIL] Failed to load {name}: {str(e)}")
        else:
            logger.warning(f"  [WARN] Not found: {name}")
    
    logger.info(f"Loaded {loaded_count}/{len(pricing_files)} pricing files")
    
    return pricing_data


# ============================================================================
# STEP 4: CREATE COST TIME SERIES
# ============================================================================

def create_cost_timeseries(
    usage_df: pd.DataFrame,
    aws_pricing_df: pd.DataFrame,
    azure_pricing: Dict,
    config: Config,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Combine usage data with pricing to create cost time series
    """
    logger.info(f"\n{'='*60}")
    logger.info("[4/6] Creating Cost Time Series")
    logger.info(f"{'='*60}")
    
    # Make a copy
    df = usage_df.copy()
    
    # Convert timestamps
    logger.info("Converting timestamps...")
    df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_numeric(df['end_time'], errors='coerce')
    
    # Remove rows with invalid timestamps
    invalid_times = df['start_time'].isna() | df['end_time'].isna()
    if invalid_times.any():
        logger.warning(f"Removing {invalid_times.sum()} rows with invalid timestamps")
        df = df[~invalid_times]
    
    # Calculate duration
    df['duration_hours'] = (df['end_time'] - df['start_time']) / (1e6 * 3600)
    
    # Filter out invalid durations
    invalid_duration = (df['duration_hours'] <= 0) | (df['duration_hours'] > 24*365)
    if invalid_duration.any():
        logger.warning(f"Removing {invalid_duration.sum()} rows with invalid duration")
        df = df[~invalid_duration]
    
    logger.info(f"Duration stats - Mean: {df['duration_hours'].mean():.2f}h, "
               f"Median: {df['duration_hours'].median():.2f}h")
    
    # Assign cloud providers
    logger.info("Assigning cloud providers...")
    df['cloud_provider'] = np.random.choice(['aws', 'azure'], size=len(df))
    provider_dist = df['cloud_provider'].value_counts()
    logger.info(f"  AWS: {provider_dist.get('aws', 0):,} ({provider_dist.get('aws', 0)/len(df)*100:.1f}%)")
    logger.info(f"  Azure: {provider_dist.get('azure', 0):,} ({provider_dist.get('azure', 0)/len(df)*100:.1f}%)")
    
    # Assign service types
    logger.info("Assigning service types...")
    df['service_type'] = np.random.choice(
        config.SERVICE_TYPES,
        size=len(df),
        p=config.SERVICE_PROBABILITIES
    )
    service_dist = df['service_type'].value_counts()
    for service_type in config.SERVICE_TYPES:
        count = service_dist.get(service_type, 0)
        logger.info(f"  {service_type}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Simulate storage and network usage
    logger.info("Simulating storage and network usage...")
    df['storage_gb'] = np.random.uniform(
        config.STORAGE_MIN_GB,
        config.STORAGE_MAX_GB,
        size=len(df)
    )
    df['network_gb'] = df['duration_hours'] * np.random.uniform(
        config.NETWORK_MIN_RATE,
        config.NETWORK_MAX_RATE,
        size=len(df)
    )
    
    logger.info(f"Storage - Mean: {df['storage_gb'].mean():.1f} GB")
    logger.info(f"Network - Mean: {df['network_gb'].mean():.1f} GB")
    
    # Calculate costs
    logger.info("Calculating costs...")
    
    def calculate_cost(row):
        """Calculate total cost for a row"""
        if row['cloud_provider'] == 'aws':
            cpu_cost = row['cpu_usage'] * config.AWS_CPU_PRICE * row['duration_hours']
            memory_cost = row['memory_usage'] * config.AWS_MEMORY_PRICE * row['duration_hours']
            storage_cost = row['storage_gb'] * config.AWS_STORAGE_PRICE * row['duration_hours']
            network_cost = row['network_gb'] * config.AWS_NETWORK_PRICE
        else:  # Azure
            cpu_cost = row['cpu_usage'] * config.AZURE_CPU_PRICE * row['duration_hours']
            memory_cost = row['memory_usage'] * config.AZURE_MEMORY_PRICE * row['duration_hours']
            storage_cost = row['storage_gb'] * config.AZURE_STORAGE_PRICE * row['duration_hours']
            network_cost = row['network_gb'] * config.AZURE_NETWORK_PRICE
        
        total = cpu_cost + memory_cost + storage_cost + network_cost
        
        # Apply service-specific adjustments
        if row['service_type'] == 'kubernetes':
            total *= config.KUBERNETES_OVERHEAD
        elif row['service_type'] == 'storage':
            total *= config.STORAGE_DISCOUNT
        
        return total
    
    df['cost'] = df.apply(calculate_cost, axis=1)
    
    logger.info(f"Cost stats - Mean: ${df['cost'].mean():.2f}, "
               f"Median: ${df['cost'].median():.2f}, "
               f"Total: ${df['cost'].sum():,.2f}")
    
    # Select relevant columns
    cost_df = df[[
        'start_time', 'end_time', 'cloud_provider', 'service_type',
        'cpu_usage', 'memory_usage', 'storage_gb', 'network_gb',
        'duration_hours', 'cost'
    ]].copy()
    
    # Aggregate by hour
    logger.info("Aggregating by hour...")
    cost_df['hour'] = (cost_df['start_time'] / (1e6 * 3600)).astype(int)
    
    hourly_costs = cost_df.groupby(['hour', 'cloud_provider', 'service_type']).agg({
        'cost': 'sum',
        'cpu_usage': 'mean',
        'memory_usage': 'mean',
        'storage_gb': 'mean',
        'network_gb': 'sum',
        'duration_hours': 'sum'
    }).reset_index()
    
    logger.info(f"Created {len(hourly_costs):,} hourly cost records")
    logger.info(f"Time range: {hourly_costs['hour'].min()} to {hourly_costs['hour'].max()}")
    
    # Save
    output_file = config.PROCESSED_DIR / 'cost_timeseries.csv'
    save_dataframe(hourly_costs, output_file, logger)
    
    return hourly_costs


# ============================================================================
# STEP 5: NORMALIZE DATA
# ============================================================================

def normalize_data(
    cost_df: pd.DataFrame,
    config: Config,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, object]:
    """
    Normalize data for ML model training with validation
    """
    logger.info(f"\n{'='*60}")
    logger.info("[5/6] Normalizing Data for ML")
    logger.info(f"{'='*60}")
    
    from sklearn.preprocessing import StandardScaler
    
    # Features to normalize
    features = ['cpu_usage', 'memory_usage', 'storage_gb', 'network_gb', 'duration_hours', 'cost']
    
    # Validate features exist
    missing_features = set(features) - set(cost_df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Check for infinite or NaN values
    for feature in features:
        if cost_df[feature].isnull().any():
            null_count = cost_df[feature].isnull().sum()
            logger.warning(f"  {feature}: {null_count} null values - filling with mean")
            cost_df[feature].fillna(cost_df[feature].mean(), inplace=True)
        
        if np.isinf(cost_df[feature]).any():
            inf_count = np.isinf(cost_df[feature]).sum()
            logger.warning(f"  {feature}: {inf_count} infinite values - replacing with max")
            cost_df[feature].replace([np.inf, -np.inf], cost_df[feature].max(), inplace=True)
    
    # Log statistics before normalization
    logger.info("Pre-normalization statistics:")
    for feature in features:
        logger.info(f"  {feature}: mean={cost_df[feature].mean():.4f}, "
                   f"std={cost_df[feature].std():.4f}, "
                   f"min={cost_df[feature].min():.4f}, "
                   f"max={cost_df[feature].max():.4f}")
    
    # Create and fit scaler
    scaler = StandardScaler()
    cost_df_normalized = cost_df.copy()
    cost_df_normalized[features] = scaler.fit_transform(cost_df[features])
    
    # Log statistics after normalization
    logger.info("Post-normalization statistics:")
    for feature in features:
        logger.info(f"  {feature}: mean={cost_df_normalized[feature].mean():.4f}, "
                   f"std={cost_df_normalized[feature].std():.4f}")
    
    # Save scaler
    scaler_path = config.MODELS_DIR / 'scaler.pkl'
    try:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"[OK] Scaler saved to: {scaler_path}")
    except Exception as e:
        logger.error(f"Failed to save scaler: {str(e)}")
        raise
    
    # Save normalized data
    output_file = config.PROCESSED_DIR / 'normalized_data.pkl'
    try:
        cost_df_normalized.to_pickle(output_file)
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        logger.info(f"[OK] Normalized data saved: {output_file.name} ({file_size_mb:.1f} MB)")
    except Exception as e:
        logger.error(f"Failed to save normalized data: {str(e)}")
        raise
    
    return cost_df_normalized, scaler


# ============================================================================
# STEP 6: GENERATE SUMMARY
# ============================================================================

def generate_summary(
    usage_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    logger: logging.Logger
) -> None:
    """
    Generate and log comprehensive summary statistics
    """
    logger.info(f"\n{'='*60}")
    logger.info("[6/6] Summary Statistics")
    logger.info(f"{'='*60}")
    
    # Usage data summary
    logger.info("\n[DATA] USAGE DATA:")
    logger.info(f"  Total records: {len(usage_df):,}")
    logger.info(f"  Date range: {usage_df['start_time'].min()} to {usage_df['end_time'].max()}")
    logger.info(f"  CPU usage - Mean: {usage_df['cpu_usage'].mean():.4f}, "
               f"Median: {usage_df['cpu_usage'].median():.4f}")
    logger.info(f"  Memory usage - Mean: {usage_df['memory_usage'].mean():.4f}, "
               f"Median: {usage_df['memory_usage'].median():.4f}")
    
    if 'storage_gb' in usage_df.columns:
        logger.info(f"  Storage - Mean: {usage_df['storage_gb'].mean():.2f} GB, "
                   f"Median: {usage_df['storage_gb'].median():.2f} GB")
    if 'network_gb' in usage_df.columns:
        logger.info(f"  Network - Mean: {usage_df['network_gb'].mean():.2f} GB, "
                   f"Median: {usage_df['network_gb'].median():.2f} GB")
    
    # Cost data summary
    logger.info("\n[COST] COST DATA:")
    logger.info(f"  Total records: {len(cost_df):,}")
    logger.info(f"  Total cost: ${cost_df['cost'].sum():,.2f}")
    logger.info(f"  Mean cost/hour: ${cost_df['cost'].mean():.2f}")
    logger.info(f"  Median cost/hour: ${cost_df['cost'].median():.2f}")
    logger.info(f"  Max cost/hour: ${cost_df['cost'].max():.2f}")
    logger.info(f"  Min cost/hour: ${cost_df['cost'].min():.2f}")
    
    # Provider breakdown
    logger.info("\n[CLOUD]  BY CLOUD PROVIDER:")
    provider_stats = cost_df.groupby('cloud_provider')['cost'].agg(['sum', 'mean', 'count'])
    for provider, row in provider_stats.iterrows():
        logger.info(f"  {provider.upper()}:")
        logger.info(f"    Total: ${row['sum']:,.2f}")
        logger.info(f"    Average: ${row['mean']:.2f}")
        logger.info(f"    Count: {int(row['count']):,}")
    
    # Service type breakdown
    logger.info("\n[SERVICE] BY SERVICE TYPE:")
    service_stats = cost_df.groupby('service_type')['cost'].agg(['sum', 'mean', 'count'])
    for service, row in service_stats.iterrows():
        logger.info(f"  {service.upper()}:")
        logger.info(f"    Total: ${row['sum']:,.2f}")
        logger.info(f"    Average: ${row['mean']:.2f}")
        logger.info(f"    Count: {int(row['count']):,}")
    
    logger.info(f"\n{'='*60}")
    logger.info("[SUCCESS] PREPROCESSING COMPLETE!")
    logger.info(f"{'='*60}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main preprocessing pipeline"""
    
    # Initialize configuration
    config = Config()
    
    # Setup logging
    logger = setup_logging(config)
    
    logger.info("="*60)
    logger.info("CLOUD COST DATA PREPROCESSING PIPELINE")
    logger.info("="*60)
    logger.info(f"Raw data directory: {config.RAW_DIR}")
    logger.info(f"Processed data directory: {config.PROCESSED_DIR}")
    logger.info(f"Models directory: {config.MODELS_DIR}")
    logger.info(f"Logs directory: {config.LOGS_DIR}")
    
    try:
        # Define file paths
        google_trace_file = config.RAW_DIR / 'workloads' / 'instance_usage-000000000000.json'
        aws_ec2_file = config.RAW_DIR / 'pricing' / 'aws_ec2_pricing.json'
        google_sample_file = config.PROCESSED_DIR / 'google_trace_sample.csv'
        aws_sample_file = config.PROCESSED_DIR / 'aws_ec2_sample.csv'
        
        # Step 1: Sample Google trace
        usage_df = sample_google_trace(
            google_trace_file,
            google_sample_file,
            config,
            logger
        )
        
        # Step 2: Sample AWS pricing
        aws_pricing_df = sample_aws_pricing(
            aws_ec2_file,
            aws_sample_file,
            config,
            logger
        )
        
        # Step 3: Load small pricing files
        azure_pricing = load_small_pricing_files(config, logger)
        
        # Step 4: Create cost time series
        cost_df = create_cost_timeseries(
            usage_df,
            aws_pricing_df,
            azure_pricing,
            config,
            logger
        )
        
        # Step 5: Normalize for ML
        normalized_df, scaler = normalize_data(cost_df, config, logger)
        
        # Step 6: Generate summary
        generate_summary(usage_df, cost_df, logger)
        
        logger.info(f"[FILES] Processed files location: {config.PROCESSED_DIR}")
        logger.info(f"[FILES] Model artifacts location: {config.MODELS_DIR}")
        logger.info(f"[LOG] Log files location: {config.LOGS_DIR}")
        
        return 0  # Success
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)