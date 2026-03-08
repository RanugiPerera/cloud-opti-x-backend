"""
Enhanced Cloud Cost Data Preprocessing Pipeline
Processes Google Cluster Trace and Cloud Pricing data for ML model training

Key fixes in this version:
  1. Google Cluster Trace utilization fractions scaled to realistic vCPU/GB-RAM
     counts — pushes costs into meaningful $5-$500/hr range
  2. Synthetic cost capped at 95th percentile of real data — prevents outlier
     distribution mismatch that caused mean collapse during training
  3. Real AWS/Azure pricing parsed from downloaded files
  4. Dataset extended to TARGET_HOURLY_ROWS via pattern-preserving synthesis
  5. No log1p in preprocessing — training script handles it
"""

import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, Tuple
from dataclasses import dataclass, field
import sys


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:

    # Sampling
    GOOGLE_SAMPLE_SIZE: int = 500_000
    AWS_SAMPLE_SIZE:    int = 50_000
    PROGRESS_INTERVAL:  int = 1_000_000

    # -----------------------------------------------------------------------
    # Scaling factors for Google Cluster Trace
    # The trace stores utilization as fractions of one machine (0.0 – 1.0).
    # We scale to a realistic multi-VM fleet so pricing produces meaningful $.
    #   CPU_SCALE = 64  →  max ~64 vCPUs  (typical large instance fleet)
    #   MEM_SCALE = 256 →  max ~256 GB RAM
    # -----------------------------------------------------------------------
    CPU_SCALE: float = 64.0
    MEM_SCALE: float = 256.0

    # Fallback pricing (used only if downloaded files are missing)
    AWS_CPU_PRICE:     float = 0.05
    AWS_MEMORY_PRICE:  float = 0.02
    AWS_STORAGE_PRICE: float = 0.1 / 730
    AWS_NETWORK_PRICE: float = 0.09

    AZURE_CPU_PRICE:     float = 0.055
    AZURE_MEMORY_PRICE:  float = 0.022
    AZURE_STORAGE_PRICE: float = 0.12 / 730
    AZURE_NETWORK_PRICE: float = 0.087

    # Resource simulation
    STORAGE_MIN_GB:   float = 10
    STORAGE_MAX_GB:   float = 500
    NETWORK_MIN_RATE: float = 0.5
    NETWORK_MAX_RATE: float = 5.0

    SERVICE_TYPES:         list = field(default_factory=lambda: ['vm', 'storage', 'network', 'kubernetes'])
    SERVICE_PROBABILITIES: list = field(default_factory=lambda: [0.5, 0.2, 0.2, 0.1])

    KUBERNETES_OVERHEAD: float = 1.3
    STORAGE_DISCOUNT:    float = 0.8

    # Target rows in final hourly CSV (real + synthetic)
    TARGET_HOURLY_ROWS: int = 4000

    def __post_init__(self):
        self.BASE_DIR      = Path(__file__).parent.parent
        self.RAW_DIR       = self.BASE_DIR / 'data' / 'raw'
        self.PROCESSED_DIR = self.BASE_DIR / 'data' / 'processed'
        self.MODELS_DIR    = self.BASE_DIR / 'trained_models'
        self.LOGS_DIR      = self.BASE_DIR / 'logs'

        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(config: Config) -> logging.Logger:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = config.LOGS_DIR / f'preprocessing_{timestamp}.log'

    logger = logging.getLogger('preprocessing')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"Logging initialized -> {log_file}")
    return logger


# ============================================================================
# UTILITIES
# ============================================================================

def validate_file_exists(filepath: Path, logger: logging.Logger) -> bool:
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return False
    size_mb = filepath.stat().st_size / (1024 * 1024)
    logger.info(f"Found: {filepath.name} ({size_mb:.1f} MB)")
    return True


def save_dataframe(df: pd.DataFrame, filepath: Path, logger: logging.Logger) -> None:
    try:
        df.to_csv(filepath, index=False)
        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"[OK] Saved {filepath.name} -- {len(df):,} rows, {size_mb:.1f} MB")
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {e}")
        raise


def validate_dataframe(df: pd.DataFrame, required_cols: list, logger: logging.Logger) -> bool:
    if df.empty:
        logger.error("DataFrame is empty")
        return False
    missing = set(required_cols) - set(df.columns)
    if missing:
        logger.error(f"Missing columns: {missing}")
        return False
    nulls = df[required_cols].isnull().sum()
    if nulls.any():
        logger.warning(f"Null values:\n{nulls[nulls > 0]}")
    return True


def _parse_memory_gb(memory_str: str) -> float:
    if not memory_str or str(memory_str).strip() in ('', 'NA', 'N/A'):
        return 0.0
    s     = str(memory_str).replace(',', '')
    match = re.search(r'([\d.]+)\s*(GiB|GB|MiB|MB)?', s, re.IGNORECASE)
    if not match:
        return 0.0
    val  = float(match.group(1))
    unit = (match.group(2) or 'GB').upper()
    return val / 1024 if 'M' in unit else val


def _parse_vcpu(vcpu_str: str) -> float:
    if not vcpu_str or str(vcpu_str).strip() in ('', 'NA', 'N/A'):
        return 0.0
    match = re.search(r'([\d.]+)', str(vcpu_str))
    return float(match.group(1)) if match else 0.0


# ============================================================================
# REAL PRICING EXTRACTION
# ============================================================================

def extract_aws_pricing(
    pricing_files: Dict[str, dict],
    config: Config,
    logger: logging.Logger
) -> Dict[str, float]:
    rates = {
        'cpu_price':     config.AWS_CPU_PRICE,
        'memory_price':  config.AWS_MEMORY_PRICE,
        'storage_price': config.AWS_STORAGE_PRICE,
        'network_price': config.AWS_NETWORK_PRICE,
    }

    if 'aws_ec2' in pricing_files:
        try:
            products  = pricing_files['aws_ec2'].get('products', {})
            terms     = pricing_files['aws_ec2'].get('terms', {}).get('OnDemand', {})
            cpu_rates, mem_rates = [], []

            for sku, product in products.items():
                attrs = product.get('attributes', {})
                if (attrs.get('operatingSystem') != 'Linux'
                        or attrs.get('tenancy') != 'Shared'
                        or not attrs.get('instanceType')):
                    continue
                vcpus  = _parse_vcpu(attrs.get('vcpu', ''))
                mem_gb = _parse_memory_gb(attrs.get('memory', ''))
                if vcpus <= 0 or mem_gb <= 0:
                    continue
                for term in terms.get(sku, {}).values():
                    for pd_item in term.get('priceDimensions', {}).values():
                        price = float(pd_item['pricePerUnit'].get('USD', 0))
                        if price > 0:
                            cpu_rates.append(price / vcpus)
                            mem_rates.append(price / mem_gb)

            if cpu_rates:
                rates['cpu_price'] = float(np.median(cpu_rates))
                logger.info(f"  AWS CPU  ${rates['cpu_price']:.4f}/vCPU/hr "
                            f"(median of {len(cpu_rates)} types)")
            if mem_rates:
                rates['memory_price'] = float(np.median(mem_rates))
                logger.info(f"  AWS Mem  ${rates['memory_price']:.4f}/GB/hr")

        except Exception as e:
            logger.warning(f"  AWS EC2 parse failed: {e} -- using fallback")

    if 'aws_s3' in pricing_files:
        try:
            products = pricing_files['aws_s3'].get('products', {})
            terms    = pricing_files['aws_s3'].get('terms', {}).get('OnDemand', {})
            s_rates  = []
            for sku, product in products.items():
                if 'TimedStorage' not in product.get('attributes', {}).get('usagetype', ''):
                    continue
                for term in terms.get(sku, {}).values():
                    for pd_item in term.get('priceDimensions', {}).values():
                        price = float(pd_item['pricePerUnit'].get('USD', 0))
                        if 0 < price < 1:
                            s_rates.append(price / 730)
            if s_rates:
                rates['storage_price'] = float(np.median(s_rates))
                logger.info(f"  AWS Storage ${rates['storage_price']:.6f}/GB/hr")
        except Exception as e:
            logger.warning(f"  AWS S3 parse failed: {e} -- using fallback")

    if 'aws_vpc' in pricing_files:
        try:
            products = pricing_files['aws_vpc'].get('products', {})
            terms    = pricing_files['aws_vpc'].get('terms', {}).get('OnDemand', {})
            n_rates  = []
            for sku, product in products.items():
                if 'DataTransfer' not in product.get('attributes', {}).get('usagetype', ''):
                    continue
                for term in terms.get(sku, {}).values():
                    for pd_item in term.get('priceDimensions', {}).values():
                        price = float(pd_item['pricePerUnit'].get('USD', 0))
                        if 0 < price < 1:
                            n_rates.append(price)
            if n_rates:
                rates['network_price'] = float(np.median(n_rates))
                logger.info(f"  AWS Network ${rates['network_price']:.4f}/GB")
        except Exception as e:
            logger.warning(f"  AWS VPC parse failed: {e} -- using fallback")

    return rates


def extract_azure_pricing(
    pricing_files: Dict[str, dict],
    config: Config,
    logger: logging.Logger
) -> Dict[str, float]:
    rates = {
        'cpu_price':     config.AZURE_CPU_PRICE,
        'memory_price':  config.AZURE_MEMORY_PRICE,
        'storage_price': config.AZURE_STORAGE_PRICE,
        'network_price': config.AZURE_NETWORK_PRICE,
    }

    if 'azure_vm' in pricing_files:
        try:
            items = pricing_files['azure_vm']
            if isinstance(items, dict):
                items = items.get('Items', [])
            cpu_rates, mem_rates = [], []
            for item in items:
                if item.get('type') != 'Consumption':
                    continue
                price = float(item.get('retailPrice', 0))
                if price <= 0:
                    continue
                sku = item.get('armSkuName', '')
                m   = re.search(r'[A-Za-z](\d+)[a-z_]', sku)
                if m:
                    vcpus = float(m.group(1))
                    if vcpus > 0:
                        cpu_rates.append(price / vcpus)
                        mem_rates.append(price / (vcpus * 6))
            if cpu_rates:
                rates['cpu_price'] = float(np.median(cpu_rates))
                logger.info(f"  Azure CPU  ${rates['cpu_price']:.4f}/vCPU/hr "
                            f"(median of {len(cpu_rates)} SKUs)")
            if mem_rates:
                rates['memory_price'] = float(np.median(mem_rates))
                logger.info(f"  Azure Mem  ${rates['memory_price']:.4f}/GB/hr")
        except Exception as e:
            logger.warning(f"  Azure VM parse failed: {e} -- using fallback")

    if 'azure_storage' in pricing_files:
        try:
            items = pricing_files['azure_storage']
            if isinstance(items, dict):
                items = items.get('Items', [])
            s_rates = []
            for item in items:
                if 'LRS' in item.get('skuName', '') and item.get('type') == 'Consumption':
                    price = float(item.get('retailPrice', 0))
                    if 0 < price < 1:
                        s_rates.append(price / 730)
            if s_rates:
                rates['storage_price'] = float(np.median(s_rates))
                logger.info(f"  Azure Storage ${rates['storage_price']:.6f}/GB/hr")
        except Exception as e:
            logger.warning(f"  Azure Storage parse failed: {e} -- using fallback")

    if 'azure_vnet' in pricing_files:
        try:
            items = pricing_files['azure_vnet']
            if isinstance(items, dict):
                items = items.get('Items', [])
            n_rates = []
            for item in items:
                price = float(item.get('retailPrice', 0))
                if 0 < price < 1 and 'Transfer' in item.get('meterName', ''):
                    n_rates.append(price)
            if n_rates:
                rates['network_price'] = float(np.median(n_rates))
                logger.info(f"  Azure Network ${rates['network_price']:.4f}/GB")
        except Exception as e:
            logger.warning(f"  Azure VNet parse failed: {e} -- using fallback")

    return rates


# ============================================================================
# STEP 1: SAMPLE GOOGLE CLUSTER TRACE
# ============================================================================

def sample_google_trace(
    input_file: Path,
    output_file: Path,
    config: Config,
    logger: logging.Logger
) -> pd.DataFrame:

    logger.info(f"\n{'='*60}")
    logger.info("[1/5] Sampling Google Cluster Trace")
    logger.info(f"{'='*60}")

    if not validate_file_exists(input_file, logger):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logger.info("Counting total lines...")
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    logger.info(f"Total lines: {total_lines:,}")

    sample_rate = min(1.0, config.GOOGLE_SAMPLE_SIZE / total_lines) if total_lines > 0 else 0
    logger.info(f"Sampling rate: {sample_rate:.2%}")

    data, parse_errors = [], 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if (i + 1) % config.PROGRESS_INTERVAL == 0:
                logger.info(f"  {i+1:,} lines | {len(data):,} samples | {parse_errors} errors")
            if np.random.random() < sample_rate:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
            if len(data) >= config.GOOGLE_SAMPLE_SIZE:
                logger.info(f"  Target reached at line {i+1:,}")
                break

    if not data:
        raise ValueError("No valid data collected from input file")

    df = pd.DataFrame(data)

    relevant_cols = ['start_time', 'end_time', 'collection_id',
                     'instance_index', 'machine_id', 'average_usage']
    df = df[[c for c in relevant_cols if c in df.columns]]

    if 'average_usage' in df.columns and isinstance(df['average_usage'].iloc[0], dict):
        logger.info("Extracting nested CPU and memory usage...")
        df['cpu_usage']    = df['average_usage'].apply(
            lambda x: x.get('cpus', 0) if isinstance(x, dict) else 0)
        df['memory_usage'] = df['average_usage'].apply(
            lambda x: x.get('memory', 0) if isinstance(x, dict) else 0)
        df = df.drop('average_usage', axis=1)

    # ------------------------------------------------------------------
    # CRITICAL FIX: Scale utilization fractions to realistic resource counts
    #
    # Google Cluster Trace stores values as fractions of one machine:
    #   cpu_usage = 0.006 means 0.6% of one machine's CPU
    #
    # Without scaling, costs are ~$0.001/hr — too small for meaningful
    # forecasting. Scaling to a 64-vCPU / 256-GB fleet pushes costs
    # into the $5-$500/hr range with real variance the model can learn.
    # ------------------------------------------------------------------
    logger.info(f"Scaling utilization fractions to realistic fleet size:")
    logger.info(f"  CPU:    x{config.CPU_SCALE} (0-1 fraction -> 0-{config.CPU_SCALE:.0f} vCPUs)")
    logger.info(f"  Memory: x{config.MEM_SCALE} (0-1 fraction -> 0-{config.MEM_SCALE:.0f} GB)")

    df['cpu_usage']    = df['cpu_usage'].clip(0, 1) * config.CPU_SCALE
    df['memory_usage'] = df['memory_usage'].clip(0, 1) * config.MEM_SCALE

    logger.info(f"CPU    -- Mean: {df['cpu_usage'].mean():.2f} vCPUs, "
                f"Max: {df['cpu_usage'].max():.2f} vCPUs")
    logger.info(f"Memory -- Mean: {df['memory_usage'].mean():.2f} GB, "
                f"Max: {df['memory_usage'].max():.2f} GB")

    if not validate_dataframe(df, ['start_time', 'end_time', 'cpu_usage', 'memory_usage'], logger):
        raise ValueError("DataFrame validation failed")

    save_dataframe(df, output_file, logger)
    return df


# ============================================================================
# STEP 2: LOAD PRICING FILES
# ============================================================================

def load_pricing_files(config: Config, logger: logging.Logger) -> Dict[str, dict]:

    logger.info(f"\n{'='*60}")
    logger.info("[2/5] Loading Pricing Files")
    logger.info(f"{'='*60}")

    pricing_files = {
        'azure_vm':      config.RAW_DIR / 'pricing' / 'azure_vm_pricing.json',
        'azure_storage': config.RAW_DIR / 'pricing' / 'azure_storage_pricing.json',
        'azure_vnet':    config.RAW_DIR / 'pricing' / 'azure_vnet_pricing.json',
        'azure_aks':     config.RAW_DIR / 'pricing' / 'azure_aks_pricing.json',
        'aws_ec2':       config.RAW_DIR / 'pricing' / 'aws_ec2_pricing.json',
        'aws_s3':        config.RAW_DIR / 'pricing' / 'aws_s3_pricing.json',
        'aws_vpc':       config.RAW_DIR / 'pricing' / 'aws_vpc_pricing.json',
        'aws_eks':       config.RAW_DIR / 'pricing' / 'aws_eks_pricing.json',
    }

    loaded = {}
    for name, path in pricing_files.items():
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    loaded[name] = json.load(f)
                size_mb = path.stat().st_size / (1024 * 1024)
                logger.info(f"  [OK] {name} ({size_mb:.1f} MB)")
            except Exception as e:
                logger.warning(f"  [FAIL] {name}: {e}")
        else:
            logger.warning(f"  [MISSING] {name} -- will use fallback pricing")

    logger.info(f"Loaded {len(loaded)}/{len(pricing_files)} pricing files")
    return loaded


# ============================================================================
# STEP 3: CREATE COST TIME SERIES
# ============================================================================

def _extend_timeseries(
    hourly: pd.DataFrame,
    rows_needed: int,
    aws_rates: dict,
    azure_rates: dict,
    config: Config,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Extend hourly time series with synthetic rows that preserve
    real data patterns: diurnal cycle, weekly seasonality, variance.

    Cost capped at 95th percentile of real data to prevent extreme
    outliers creating a distribution mismatch that causes mean collapse.
    """
    tmp        = hourly.copy()
    tmp['hod'] = tmp['hour'] % 24
    diurnal    = tmp.groupby('hod')[['cpu_usage', 'memory_usage', 'cost']].mean()

    cpu_std  = hourly['cpu_usage'].std()
    mem_std  = hourly['memory_usage'].std()

    # Cap synthetic costs at 95th percentile of real distribution
    real_cost_p95 = hourly['cost'].quantile(0.95)
    real_cost_std = hourly['cost'].std()

    logger.info(f"  Synthetic cost ceiling: ${real_cost_p95:.2f} (real p95)")
    logger.info(f"  Real cost mean: ${hourly['cost'].mean():.2f}, "
                f"std: ${real_cost_std:.2f}")

    hourly_trend = 0.03 / (30 * 24)  # 3% per month
    start_hour   = int(hourly['hour'].max()) + 1

    blended_cpu     = (aws_rates['cpu_price']     + azure_rates['cpu_price'])     / 2
    blended_mem     = (aws_rates['memory_price']  + azure_rates['memory_price'])  / 2
    blended_storage = (aws_rates['storage_price'] + azure_rates['storage_price']) / 2
    blended_net     = (aws_rates['network_price'] + azure_rates['network_price']) / 2

    rows = []
    for i in range(rows_needed):
        h              = start_hour + i
        hod            = h % 24
        dow            = (h // 24) % 7
        trend_mult     = 1.0 + hourly_trend * i
        weekend_factor = 0.80 if dow >= 5 else 1.0

        cpu = max(0.0,
                  diurnal.loc[hod, 'cpu_usage'] * trend_mult * weekend_factor
                  + np.random.normal(0, cpu_std * 0.3))
        mem = max(0.0,
                  diurnal.loc[hod, 'memory_usage'] * trend_mult * weekend_factor
                  + np.random.normal(0, mem_std * 0.3))

        storage  = np.random.uniform(config.STORAGE_MIN_GB, config.STORAGE_MAX_GB)
        duration = max(0.1, np.random.exponential(2.0))
        network  = duration * np.random.uniform(
            config.NETWORK_MIN_RATE, config.NETWORK_MAX_RATE)

        cost = (
            (cpu * blended_cpu + mem * blended_mem + storage * blended_storage)
            * duration + network * blended_net
        )
        # Small noise, then hard cap at real p95 — no synthetic outliers
        cost = cost * trend_mult + np.random.normal(0, real_cost_std * 0.05)
        cost = float(np.clip(cost, 0.001, real_cost_p95))

        rows.append({
            'hour':           h,
            'cpu_usage':      cpu,
            'memory_usage':   mem,
            'storage_gb':     storage,
            'network_gb':     network,
            'duration_hours': duration,
            'cost':           cost,
        })

    synth_df = pd.DataFrame(rows)
    combined = pd.concat([hourly, synth_df], ignore_index=True)

    logger.info(f"  Real rows:      {len(hourly):,}")
    logger.info(f"  Synthetic rows: {len(synth_df):,}")
    logger.info(f"  Combined total: {len(combined):,}")
    logger.info(f"  Synth cost:     ${synth_df['cost'].min():.2f} - "
                f"${synth_df['cost'].max():.2f}")
    return combined


def create_cost_timeseries(
    usage_df: pd.DataFrame,
    pricing_files: Dict[str, dict],
    config: Config,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    logger.info(f"\n{'='*60}")
    logger.info("[3/5] Creating Cost Time Series")
    logger.info(f"{'='*60}")

    logger.info("\nExtracting AWS pricing...")
    aws_rates   = extract_aws_pricing(pricing_files, config, logger)

    logger.info("\nExtracting Azure pricing...")
    azure_rates = extract_azure_pricing(pricing_files, config, logger)

    logger.info("\nPricing rates in use:")
    logger.info(f"  AWS   CPU ${aws_rates['cpu_price']:.4f}/vCPU/hr | "
                f"Mem ${aws_rates['memory_price']:.4f}/GB/hr | "
                f"Storage ${aws_rates['storage_price']:.6f}/GB/hr | "
                f"Net ${aws_rates['network_price']:.4f}/GB")
    logger.info(f"  Azure CPU ${azure_rates['cpu_price']:.4f}/vCPU/hr | "
                f"Mem ${azure_rates['memory_price']:.4f}/GB/hr | "
                f"Storage ${azure_rates['storage_price']:.6f}/GB/hr | "
                f"Net ${azure_rates['network_price']:.4f}/GB")

    df = usage_df.copy()
    df['start_time'] = pd.to_numeric(df['start_time'], errors='coerce')
    df['end_time']   = pd.to_numeric(df['end_time'],   errors='coerce')
    df = df.dropna(subset=['start_time', 'end_time'])
    df['duration_hours'] = (df['end_time'] - df['start_time']) / (1e6 * 3600)
    df = df[df['duration_hours'] > 0]

    df['cloud_provider'] = np.random.choice(['aws', 'azure'], size=len(df))
    df['service_type']   = np.random.choice(
        config.SERVICE_TYPES, size=len(df), p=config.SERVICE_PROBABILITIES)
    df['storage_gb'] = np.random.uniform(
        config.STORAGE_MIN_GB, config.STORAGE_MAX_GB, size=len(df))
    df['network_gb'] = df['duration_hours'] * np.random.uniform(
        config.NETWORK_MIN_RATE, config.NETWORK_MAX_RATE, size=len(df))

    # Vectorized cost calculation using real pricing
    # cpu_usage and memory_usage are now in vCPUs and GB (after scaling)
    is_aws        = df['cloud_provider'] == 'aws'
    cpu_price     = np.where(is_aws, aws_rates['cpu_price'],     azure_rates['cpu_price'])
    mem_price     = np.where(is_aws, aws_rates['memory_price'],  azure_rates['memory_price'])
    storage_price = np.where(is_aws, aws_rates['storage_price'], azure_rates['storage_price'])
    net_price     = np.where(is_aws, aws_rates['network_price'], azure_rates['network_price'])

    compute_cost = (
        df['cpu_usage']    * cpu_price +
        df['memory_usage'] * mem_price +
        df['storage_gb']   * storage_price
    ) * df['duration_hours'] + df['network_gb'] * net_price

    multiplier = np.ones(len(df))
    multiplier = np.where(df['service_type'] == 'kubernetes',
                          config.KUBERNETES_OVERHEAD, multiplier)
    multiplier = np.where(df['service_type'] == 'storage',
                          config.STORAGE_DISCOUNT, multiplier)
    df['cost'] = compute_cost * multiplier

    # Stats before aggregation
    provider_stats = df.groupby('cloud_provider')['cost'].agg(['sum', 'mean', 'count'])
    service_stats  = df.groupby('service_type')['cost'].agg(['sum', 'mean', 'count'])

    # Hourly aggregation
    df['hour'] = (df['start_time'] / (1e6 * 3600)).astype(int)
    hourly = df.groupby('hour').agg(
        cpu_usage=('cpu_usage',         'mean'),
        memory_usage=('memory_usage',   'mean'),
        storage_gb=('storage_gb',       'mean'),
        network_gb=('network_gb',       'sum'),
        duration_hours=('duration_hours', 'sum'),
        cost=('cost', 'sum'),
    ).reset_index()

    real_rows = len(hourly)
    logger.info(f"\nReal trace: {real_rows:,} hourly rows")
    logger.info(f"Real cost range: ${hourly['cost'].min():.2f} - "
                f"${hourly['cost'].max():.2f}")
    logger.info(f"Real cost mean:  ${hourly['cost'].mean():.2f}, "
                f"median: ${hourly['cost'].median():.2f}")

    rows_needed = config.TARGET_HOURLY_ROWS - real_rows
    if rows_needed > 0:
        logger.info(f"\nExtending by {rows_needed:,} synthetic rows...")
        hourly = _extend_timeseries(
            hourly, rows_needed, aws_rates, azure_rates, config, logger)
    else:
        logger.info(f"No extension needed ({real_rows:,} >= target)")

    # Calendar features — no log1p here, training script handles it
    hourly = hourly.sort_values('hour').reset_index(drop=True)
    hourly['hour_of_day'] = hourly['hour'] % 24
    hourly['hour_sin']    = np.sin(2 * np.pi * hourly['hour_of_day'] / 24)
    hourly['hour_cos']    = np.cos(2 * np.pi * hourly['hour_of_day'] / 24)
    hourly['day_of_week'] = (hourly['hour'] // 24) % 7
    hourly['dow_sin']     = np.sin(2 * np.pi * hourly['day_of_week'] / 7)
    hourly['dow_cos']     = np.cos(2 * np.pi * hourly['day_of_week'] / 7)
    hourly.drop(columns=['hour_of_day', 'day_of_week'], inplace=True)

    logger.info(f"\nFinal dataset: {len(hourly):,} rows")
    logger.info(f"Final cost range: ${hourly['cost'].min():.2f} - "
                f"${hourly['cost'].max():.2f}")
    logger.info(f"Final cost mean:  ${hourly['cost'].mean():.2f}, "
                f"median: ${hourly['cost'].median():.2f}")

    save_dataframe(hourly, config.PROCESSED_DIR / 'cost_timeseries.csv', logger)
    return hourly, provider_stats, service_stats


# ============================================================================
# STEP 4: AWS EC2 SAMPLE
# ============================================================================

def save_aws_ec2_sample(config: Config, logger: logging.Logger) -> None:
    logger.info(f"\n{'='*60}")
    logger.info("[4/5] Saving AWS EC2 Reference Sample")
    logger.info(f"{'='*60}")

    aws_ec2_file = config.RAW_DIR / 'pricing' / 'aws_ec2_pricing.json'
    output_file  = config.PROCESSED_DIR / 'aws_ec2_sample.csv'

    if not aws_ec2_file.exists():
        logger.warning("aws_ec2_pricing.json not found -- skipping")
        return

    try:
        with open(aws_ec2_file, 'r', encoding='utf-8') as f:
            aws_data = json.load(f)
        products    = aws_data.get('products', {})
        sample_keys = list(products.keys())[:config.AWS_SAMPLE_SIZE]
        rows = [{'sku': k, **products[k].get('attributes', {})} for k in sample_keys]
        save_dataframe(pd.DataFrame(rows), output_file, logger)
    except Exception as e:
        logger.warning(f"Failed to save AWS EC2 sample: {e}")


# ============================================================================
# STEP 5: SUMMARY
# ============================================================================

def generate_summary(
    usage_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    provider_stats: pd.DataFrame,
    service_stats: pd.DataFrame,
    logger: logging.Logger
) -> None:

    logger.info(f"\n{'='*60}")
    logger.info("[5/5] Summary Statistics")
    logger.info(f"{'='*60}")

    logger.info(f"\n[DATA] Usage records: {len(usage_df):,}")
    logger.info(f"  CPU    -- Mean: {usage_df['cpu_usage'].mean():.2f} vCPUs, "
                f"Max: {usage_df['cpu_usage'].max():.2f} vCPUs")
    logger.info(f"  Memory -- Mean: {usage_df['memory_usage'].mean():.2f} GB, "
                f"Max: {usage_df['memory_usage'].max():.2f} GB")

    logger.info(f"\n[COST] Hourly cost (raw $):")
    logger.info(f"  Rows:   {len(cost_df):,}")
    logger.info(f"  Total:  ${cost_df['cost'].sum():,.2f}")
    logger.info(f"  Mean:   ${cost_df['cost'].mean():.2f}/hr")
    logger.info(f"  Median: ${cost_df['cost'].median():.2f}/hr")
    logger.info(f"  Max:    ${cost_df['cost'].max():.2f}/hr")
    logger.info(f"  Min:    ${cost_df['cost'].min():.4f}/hr")
    logger.info(f"  Std:    ${cost_df['cost'].std():.2f}")

    logger.info(f"\n[CLOUD] By provider (pre-aggregation):")
    for provider, row in provider_stats.iterrows():
        logger.info(f"  {provider.upper()} -- Total: ${row['sum']:,.2f} | "
                    f"Mean: ${row['mean']:.2f} | Count: {int(row['count']):,}")

    logger.info(f"\n[SERVICE] By type (pre-aggregation):")
    for service, row in service_stats.iterrows():
        logger.info(f"  {service.upper()} -- Total: ${row['sum']:,.2f} | "
                    f"Mean: ${row['mean']:.2f} | Count: {int(row['count']):,}")

    logger.info(f"\n{'='*60}")
    logger.info("[SUCCESS] PREPROCESSING COMPLETE!")
    logger.info(f"{'='*60}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = Config()
    logger = setup_logging(config)

    logger.info("=" * 60)
    logger.info("CLOUD COST DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Raw data:  {config.RAW_DIR}")
    logger.info(f"Processed: {config.PROCESSED_DIR}")
    logger.info(f"CPU scale: x{config.CPU_SCALE} | Memory scale: x{config.MEM_SCALE}")

    try:
        google_trace_file  = config.RAW_DIR / 'workloads' / 'instance_usage-000000000000.json'
        google_sample_file = config.PROCESSED_DIR / 'google_trace_sample.csv'

        usage_df      = sample_google_trace(google_trace_file, google_sample_file, config, logger)
        pricing_files = load_pricing_files(config, logger)

        cost_df, provider_stats, service_stats = create_cost_timeseries(
            usage_df, pricing_files, config, logger)

        save_aws_ec2_sample(config, logger)
        generate_summary(usage_df, cost_df, provider_stats, service_stats, logger)

        logger.info(f"[FILES] Processed: {config.PROCESSED_DIR}")
        logger.info(f"[FILES] Models:    {config.MODELS_DIR}")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())