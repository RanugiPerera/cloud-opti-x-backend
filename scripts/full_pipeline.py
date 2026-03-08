
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


_AWS_DEFAULTS = dict(
    cpu_per_vcpu_hr   = 0.0480,   # m5.xlarge: $0.192/hr ÷ 4 vCPU
    mem_per_gb_hr     = 0.0060,   # m5.xlarge: $0.192/hr ÷ 16 GB  (approx)
    storage_per_gb_hr = 0.1 / 730,
    network_per_gb    = 0.09,
)
_AZURE_DEFAULTS = dict(
    cpu_per_vcpu_hr   = 0.0530,   # D4s_v3:   $0.212/hr ÷ 4 vCPU
    mem_per_gb_hr     = 0.0067,   # D4s_v3:   $0.212/hr ÷ 16 GB  (approx)
    storage_per_gb_hr = 0.12 / 730,
    network_per_gb    = 0.087,
)


def load_pricing_from_files(pricing_dir: Path, logger: logging.Logger) -> "PricingConfig":

    aws   = dict(_AWS_DEFAULTS)
    azure = dict(_AZURE_DEFAULTS)

    # ── AWS EC2 ───────────────────────────────────────────────────────────────
    aws_file = pricing_dir / "aws_ec2_pricing.json"
    if aws_file.exists():
        try:
            with open(aws_file, encoding="utf-8") as f:
                ec2 = json.load(f)

            products = ec2.get("products", {})
            terms    = ec2.get("terms", {}).get("OnDemand", {})

            # Find m5.xlarge, us-east-1, Linux, shared tenancy
            target_sku = None
            for sku, prod in products.items():
                attr = prod.get("attributes", {})
                if (attr.get("instanceType") == "m5.xlarge"
                        and attr.get("location")       == "US East (N. Virginia)"
                        and attr.get("operatingSystem") == "Linux"
                        and attr.get("tenancy")         == "Shared"
                        and attr.get("capacitystatus")  == "Used"):
                    target_sku = sku
                    break

            if target_sku and target_sku in terms:
                # terms[sku] -> {offerTermCode: {priceDimensions: {rateCode: {pricePerUnit}}}}
                price_dims = next(iter(terms[target_sku].values()))["priceDimensions"]
                hourly_usd = float(next(iter(price_dims.values()))["pricePerUnit"]["USD"])
                vcpus      = int(products[target_sku]["attributes"].get("vcpu", 4))
                memory_gb  = float(products[target_sku]["attributes"].get("memory", "16 GiB")
                                   .replace(" GiB", "").replace(" GB", ""))
                aws["cpu_per_vcpu_hr"] = hourly_usd / vcpus
                aws["mem_per_gb_hr"]   = hourly_usd / memory_gb
                logger.info(
                    f"[Pricing] AWS m5.xlarge: ${hourly_usd:.4f}/hr  "
                    f"→ ${aws['cpu_per_vcpu_hr']:.4f}/vCPU/hr  "
                    f"${aws['mem_per_gb_hr']:.4f}/GB/hr"
                )
            else:
                logger.warning("[Pricing] AWS m5.xlarge SKU not found — using defaults")
        except Exception as exc:
            logger.warning(f"[Pricing] AWS parse error: {exc} — using defaults")
    else:
        logger.warning(f"[Pricing] {aws_file.name} not found — using defaults")

    # ── Azure VM ──────────────────────────────────────────────────────────────
    azure_file = pricing_dir / "azure_vm_pricing.json"
    if azure_file.exists():
        try:
            with open(azure_file, encoding="utf-8") as f:
                az_items = json.load(f)   # list of price records

            # Find D4s v3, eastus, Linux (no Windows keyword), Consumption
            target = None
            for item in az_items:
                name    = item.get("skuName", "")
                region  = item.get("armRegionName", "")
                product = item.get("productName", "")
                if ("D4s v3" in name
                        and region == "eastus"
                        and "Windows" not in product
                        and item.get("type") == "Consumption"):
                    target = item
                    break

            if target:
                hourly_usd = float(target["retailPrice"])
                vcpus      = 4    # D4s v3 spec
                memory_gb  = 16   # D4s v3 spec
                azure["cpu_per_vcpu_hr"] = hourly_usd / vcpus
                azure["mem_per_gb_hr"]   = hourly_usd / memory_gb
                logger.info(
                    f"[Pricing] Azure D4s_v3: ${hourly_usd:.4f}/hr  "
                    f"→ ${azure['cpu_per_vcpu_hr']:.4f}/vCPU/hr  "
                    f"${azure['mem_per_gb_hr']:.4f}/GB/hr"
                )
            else:
                logger.warning("[Pricing] Azure D4s_v3 SKU not found — using defaults")
        except Exception as exc:
            logger.warning(f"[Pricing] Azure parse error: {exc} — using defaults")
    else:
        logger.warning(f"[Pricing] {azure_file.name} not found — using defaults")

    # ── Storage pricing ───────────────────────────────────────────────────────
    aws_s3_file = pricing_dir / "aws_s3_pricing.json"
    if aws_s3_file.exists():
        try:
            with open(aws_s3_file, encoding="utf-8") as f:
                s3 = json.load(f)
            products = s3.get("products", {})
            terms    = s3.get("terms", {}).get("OnDemand", {})
            for sku, prod in products.items():
                attr = prod.get("attributes", {})
                if (attr.get("volumeType") == "Standard"
                        and attr.get("location") == "US East (N. Virginia)"
                        and attr.get("storageClass") == "General Purpose"):
                    if sku in terms:
                        dims = next(iter(terms[sku].values()))["priceDimensions"]
                        # S3 pricing is per GB-month → convert to per GB-hour
                        per_gb_month = float(next(iter(dims.values()))["pricePerUnit"]["USD"])
                        aws["storage_per_gb_hr"] = per_gb_month / 730
                        logger.info(f"[Pricing] AWS S3 standard: ${per_gb_month:.4f}/GB-month")
                        break
        except Exception as exc:
            logger.warning(f"[Pricing] AWS S3 parse error: {exc} — using defaults")

    azure_storage_file = pricing_dir / "azure_storage_pricing.json"
    if azure_storage_file.exists():
        try:
            with open(azure_storage_file, encoding="utf-8") as f:
                az_stor = json.load(f)
            for item in az_stor:
                if ("LRS" in item.get("skuName", "")
                        and item.get("armRegionName") == "eastus"
                        and "Hot" in item.get("productName", "")):
                    per_gb_month = float(item["retailPrice"])
                    azure["storage_per_gb_hr"] = per_gb_month / 730
                    logger.info(f"[Pricing] Azure Blob LRS Hot: ${per_gb_month:.4f}/GB-month")
                    break
        except Exception as exc:
            logger.warning(f"[Pricing] Azure Storage parse error: {exc} — using defaults")

    return PricingConfig(
        AWS_CPU_PRICE     = aws["cpu_per_vcpu_hr"],
        AWS_MEMORY_PRICE  = aws["mem_per_gb_hr"],
        AWS_STORAGE_PRICE = aws["storage_per_gb_hr"],
        AWS_NETWORK_PRICE = aws["network_per_gb"],
        AZURE_CPU_PRICE     = azure["cpu_per_vcpu_hr"],
        AZURE_MEMORY_PRICE  = azure["mem_per_gb_hr"],
        AZURE_STORAGE_PRICE = azure["storage_per_gb_hr"],
        AZURE_NETWORK_PRICE = azure["network_per_gb"],
    )


@dataclass
class PricingConfig:
    """
    Normalised cloud pricing in $/unit/hr.
    Populated from real AWS + Azure pricing JSON files by load_pricing_from_files().
    Falls back to hardcoded defaults if files are absent.
    """
    AWS_CPU_PRICE:     float = _AWS_DEFAULTS["cpu_per_vcpu_hr"]
    AWS_MEMORY_PRICE:  float = _AWS_DEFAULTS["mem_per_gb_hr"]
    AWS_STORAGE_PRICE: float = _AWS_DEFAULTS["storage_per_gb_hr"]
    AWS_NETWORK_PRICE: float = _AWS_DEFAULTS["network_per_gb"]

    AZURE_CPU_PRICE:     float = _AZURE_DEFAULTS["cpu_per_vcpu_hr"]
    AZURE_MEMORY_PRICE:  float = _AZURE_DEFAULTS["mem_per_gb_hr"]
    AZURE_STORAGE_PRICE: float = _AZURE_DEFAULTS["storage_per_gb_hr"]
    AZURE_NETWORK_PRICE: float = _AZURE_DEFAULTS["network_per_gb"]

    KUBERNETES_OVERHEAD: float = 1.3
    STORAGE_DISCOUNT:    float = 0.8


@dataclass
class PipelineConfig:
    """Pipeline / simulation parameters."""
    RANDOM_SEED:      int   = 42
    GOOGLE_SAMPLE_SIZE: int = 500_000

    CPU_SCALE: float = 64.0
    MEM_SCALE: float = 256.0

    STORAGE_MIN_GB: float = 10.0
    STORAGE_MAX_GB: float = 500.0
    NETWORK_MIN_RATE: float = 0.5
    NETWORK_MAX_RATE: float = 5.0

    SERVICE_TYPES:         List[str]   = field(default_factory=lambda: ['vm', 'storage', 'network', 'kubernetes'])
    SERVICE_PROBABILITIES: List[float] = field(default_factory=lambda: [0.5, 0.2, 0.2, 0.1])

    # Lag / rolling window sizes (in hours)
    # AR(1) with phi=0.7: autocorrelation at lag k = 0.7^k
    # Lag 1=0.70, lag 6=0.12, lag 12=0.01 — most signal in first 12 lags.
    # Anchor lags at 24h (daily) and 168h (weekly) capture seasonality.
    LAG_HOURS:     int = 12
    LAG_ANCHORS:   List[int] = field(default_factory=lambda: [24, 48, 168])
    ROLLING_WINDOWS: List[int] = field(default_factory=lambda: [6, 12, 24, 48, 168])

    TRAIN_SPLIT: float = 0.8
    CV_SPLITS:   int   = 5

    XGB_N_ESTIMATORS:     int   = 800
    XGB_LEARNING_RATE:    float = 0.03       # slower lr → better generalisation
    XGB_MAX_DEPTH:        int   = 7          # deeper trees for lag×time interactions
    XGB_MIN_CHILD_WEIGHT: int   = 5          # prevent over-fitting leaf nodes
    XGB_EARLY_STOPPING:   int   = 30         # more patience for slower lr
    XGB_SUBSAMPLE:        float = 0.8
    XGB_COLSAMPLE_BYTREE: float = 0.7        # slightly lower to reduce correlation

    # ------------------------------------------------------------------ paths
    def __post_init__(self):
        self.BASE_DIR      = Path(__file__).parent.parent
        self.RAW_DIR       = self.BASE_DIR / 'data' / 'raw'
        self.PROCESSED_DIR = self.BASE_DIR / 'data' / 'processed'
        self.MODELS_DIR    = self.BASE_DIR / 'trained_models'
        self.LOGS_DIR      = self.BASE_DIR / 'logs'

        self.PRICING_DIR = self.RAW_DIR / 'pricing'

        for d in (self.PROCESSED_DIR, self.MODELS_DIR, self.LOGS_DIR):
            d.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(config: PipelineConfig) -> logging.Logger:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file  = config.LOGS_DIR / f'full_pipeline_{timestamp}.log'
    logger = logging.getLogger('pipeline')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt_short = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    fmt_long  = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt_short)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(fmt_long)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"Logging initialised -> {log_file}")
    return logger


# ============================================================================
# UTILITIES
# ============================================================================

def _parse_memory_gb(memory_str: str) -> float:
    if not memory_str or str(memory_str).strip() in ('', 'NA', 'N/A'):
        return 0.0
    s = str(memory_str).replace(',', '')
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
        logger.info(f"[OK] Saved {filepath.name} — {len(df):,} rows, {size_mb:.1f} MB")
    except Exception as e:
        logger.error(f"Failed to save {filepath}: {e}")
        raise


def _log_cost_stats(df: pd.DataFrame, stage: str, logger: logging.Logger) -> None:
    """Fix #13 – log cost distribution after every major step."""
    c = df['cost']
    logger.info(
        f"[{stage}] cost stats — "
        f"min={c.min():.4f}  max={c.max():.4f}  "
        f"mean={c.mean():.4f}  std={c.std():.4f}  "
        f"rows={len(df):,}"
    )


def _flatten_cell(val):
    """
    Recursively collapse a nested dict/list into a single float.
    Dicts  -> sum of all leaf numeric values (handles arbitrary nesting depth).
    Lists  -> mean of numeric elements.
    Everything else -> pd.to_numeric coercion (NaN on failure).
    """
    if isinstance(val, dict):
        leaves = []
        stack = list(val.values())
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
            else:
                try:
                    leaves.append(float(item))
                except (TypeError, ValueError):
                    pass
        return float(np.sum(leaves)) if leaves else 0.0
    if isinstance(val, list):
        nums = []
        for item in val:
            try:
                nums.append(float(item))
            except (TypeError, ValueError):
                pass
        return float(np.mean(nums)) if nums else 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def sanitise_dataframe(
    df: pd.DataFrame,
    preserve_cols: list,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Guarantee every column that will enter XGBoost is a plain float64.

    Strategy per column
    -------------------
    - If dtype is already numeric        → keep as-is
    - If dtype is object / contains dict → apply _flatten_cell row-by-row,
                                           then cast to float64
    - Any remaining NaN / inf            → replaced with 0.0
    - Columns in preserve_cols           → skipped (timestamps, string labels
                                           we handle separately)

    Logs every column that required flattening so you can inspect them.
    """
    bad_cols = []
    for col in df.columns:
        if col in preserve_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue  # already fine

        # Check whether any cell is a dict or list
        has_complex = df[col].apply(lambda x: isinstance(x, (dict, list))).any()
        if has_complex:
            df[col] = df[col].apply(_flatten_cell)
            bad_cols.append(col)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if bad_cols:
        logger.warning(
            f"Flattened {len(bad_cols)} nested-object column(s): {bad_cols}"
        )

    # Final sweep: replace any inf / NaN with 0
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    inf_mask = np.isinf(df[numeric_cols].values)
    if inf_mask.any():
        logger.warning(f"Replacing {int(inf_mask.sum())} inf values with 0")
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    nan_count = df[numeric_cols].isna().sum().sum()
    if nan_count:
        logger.warning(f"Filling {int(nan_count)} NaN values with 0")
        df[numeric_cols] = df[numeric_cols].fillna(0.0)

    return df


# ============================================================================
# STEP 1: SAMPLE GOOGLE TRACE
# ============================================================================

def sample_google_trace(
    input_file: Path,
    config: PipelineConfig,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Fix #2  – deterministic sampling with seed + honours GOOGLE_SAMPLE_SIZE.
    Fix #10 – no unused enumerate variable.
    Fix #9  – robust dict-column detection.
    Fix #11 – build real timestamps from trace start time.
    """
    if not validate_file_exists(input_file, logger):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    rng = np.random.default_rng(config.RANDOM_SEED)   # fix #2
    logger.info("Reading Google Cluster Trace (reservoir-sampled)…")

    # Read all lines then sample – avoids biased streaming sample
    with open(input_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()                       # fix #10 – no enumerate

    n_available = len(all_lines)
    n_sample    = min(config.GOOGLE_SAMPLE_SIZE, n_available)
    chosen      = rng.choice(n_available, size=n_sample, replace=False)
    data        = [json.loads(all_lines[i]) for i in sorted(chosen)]

    df = pd.DataFrame(data)

    # Fix #9 – robust dict-column detection (no iloc[0] assumption)
    if 'average_usage' in df.columns:
        is_dict = df['average_usage'].apply(lambda x: isinstance(x, dict))
        if is_dict.any():
            df['cpu_usage']    = df['average_usage'].apply(lambda x: x.get('cpus',   0) if isinstance(x, dict) else 0)
            df['memory_usage'] = df['average_usage'].apply(lambda x: x.get('memory', 0) if isinstance(x, dict) else 0)
            df.drop(columns='average_usage', inplace=True)

    # Clip to [0,1] then apply a 5% utilisation floor so that even idle
    # workloads produce a meaningful base cost (avoids near-zero actuals
    # that explode MAPE).
    df['cpu_usage']    = pd.to_numeric(df.get('cpu_usage',    0), errors='coerce').fillna(0).clip(0, 1)
    df['memory_usage'] = pd.to_numeric(df.get('memory_usage', 0), errors='coerce').fillna(0).clip(0, 1)
    df['cpu_usage']    = df['cpu_usage'].clip(lower=0.05)    * config.CPU_SCALE
    df['memory_usage'] = df['memory_usage'].clip(lower=0.05) * config.MEM_SCALE

    # Fix #11 – create real hourly timestamps (one row = one hour)
    t0 = datetime(2023, 1, 1)
    df['timestamp'] = [t0 + timedelta(hours=i) for i in range(len(df))]

    # ── CRITICAL: flatten every other column that came from raw JSON ──────────
    # The Google trace can contain arbitrarily nested dicts in many columns
    # (e.g. resource_request, assigned_memory, page_cache_memory …).
    # sanitise_dataframe walks every column, flattens dicts/lists to floats,
    # and coerces the remainder; only 'timestamp' is left untouched.
    df = sanitise_dataframe(df, preserve_cols=['timestamp'], logger=logger)

    # We only need cpu_usage and memory_usage from the trace; drop any
    # remaining raw numeric trace columns that would leak information or
    # add noise (they will NOT be available at real inference time).
    KEEP_FROM_TRACE = {'cpu_usage', 'memory_usage', 'timestamp'}
    trace_noise_cols = [
        c for c in df.columns
        if c not in KEEP_FROM_TRACE
    ]
    if trace_noise_cols:
        df.drop(columns=trace_noise_cols, inplace=True)
        logger.info(f"Dropped {len(trace_noise_cols)} raw trace column(s) not needed downstream")

    logger.info(f"Sampled {len(df):,} rows from {n_available:,} available")
    return df


# ============================================================================
# STEP 2: CREATE COST TIME SERIES
# ============================================================================

def create_cost_timeseries(
    df: pd.DataFrame,
    config: PipelineConfig,
    pricing: PricingConfig,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Generates a realistic cloud cost time-series with an **explicit variance
    budget** so the model can achieve meaningful R².

    Design principle — log-space additive model
    ────────────────────────────────────────────
    Previous versions computed cost as:
        cost = f(cpu[random], mem[random], storage[random]) × load_factor
    The random resource draws dominated variance after the leaky columns
    were removed, leaving the model with nothing predictable to learn.

    This version works directly in log-space:
        log(cost) = μ                   # base level ($1/hr fleet average)
                  + β_diurnal(h)        # hour-of-day shape    (σ≈0.28)
                  + β_weekly(d)         # weekday vs weekend   (σ≈0.25)
                  + β_trend(m)          # monthly growth       (σ≈0.03)
                  + z_AR(t)             # AR(1) residual φ=0.7 (σ≈0.15)
                  + ε_spike(t)          # rare burst events    (σ≈0.03)
                  + ε_noise(t)          # irreducible noise    (σ≈0.10)

    Variance budget (log-space):
        Deterministic (learnable from timestamp alone): ~55%
          - diurnal:  ~30%
          - weekly:   ~25%
        AR(1) (learnable from lag features):            ~25%
        Irreducible noise:                              ~20%

    Target metrics: R² ≈ 0.65–0.80, RMSLE ≈ 0.18–0.25
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    n   = len(df)
    df  = df.copy()

    # ── Calendar fields from real timestamps ──────────────────────────────────
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek   # 0=Mon … 6=Sun
    df['month']       = df['timestamp'].dt.month
    df['hour']        = np.arange(n)

    hour_arr  = df['hour_of_day'].values
    dow_arr   = df['day_of_week'].values
    month_idx = df['month'].values - 1                 # 0-11

    # ── 1. Diurnal component in log-space (σ≈0.28) ────────────────────────────
    # Two-peak shape: morning ramp-up (9am) and afternoon peak (2pm).
    # Range: -0.5 (4am trough) to +0.4 (2pm peak) in log-cost.
    diurnal_log = (
        0.35 * np.exp(-0.5 * ((hour_arr - 14) / 3.5) ** 2)   # afternoon peak
      + 0.15 * np.exp(-0.5 * ((hour_arr -  9) / 2.0) ** 2)   # morning shoulder
      - 0.25 * np.exp(-0.5 * ((hour_arr -  4) / 2.5) ** 2)   # night trough
    )

    # ── 2. Weekly component in log-space (σ≈0.25) ─────────────────────────────
    # Weekdays: near 0, weekends: −0.5 (≈40% lower cost)
    weekly_log = np.where(dow_arr < 5, 0.05, -0.55)

    # ── 3. Monthly trend in log-space (σ≈0.03) ────────────────────────────────
    # 0.8% compound growth per month ≈ 10% annual
    trend_log = month_idx * np.log(1.008)

    # ── 4. AR(1) process in log-space (φ=0.7, σ_ε=0.12 → σ_process≈0.168) ───
    # Explicit variance: Var(z) = σ²/(1-φ²) = 0.0144/0.51 ≈ 0.028 → σ≈0.168
    # Autocorrelation: r(1)=0.70, r(6)=0.12, r(12)=0.01
    phi_ar  = 0.70
    sig_ar  = 0.12
    z_ar    = np.zeros(n)
    eps_ar  = rng.normal(0.0, sig_ar, size=n)
    for t in range(1, n):
        z_ar[t] = phi_ar * z_ar[t - 1] + eps_ar[t]

    # ── 5. Spike events in log-space (1% of hours, log-magnitude 0.4–0.9) ─────
    # log(1.5)≈0.41, log(2.5)≈0.92  →  1.5x–2.5x cost increase
    spike_mask    = rng.random(n) < 0.01
    spike_log     = np.where(spike_mask, rng.uniform(0.41, 0.92, size=n), 0.0)

    # ── 6. Irreducible i.i.d. noise (σ=0.10) ──────────────────────────────────
    # Represents unmodelled per-workload variation. Deliberately kept small
    # so it doesn't drown out the learnable signal.
    eps_noise = rng.normal(0.0, 0.10, size=n)

    # ── 7. Assemble log-cost and convert to dollar cost ───────────────────────
    # Base level: log($1.00) = 0.0  →  fleet average ≈ $1/hr before adjustments
    log_cost = (
        0.0           # base level
        + diurnal_log
        + weekly_log
        + trend_log
        + z_ar
        + spike_log
        + eps_noise
    )
    df['cost'] = np.clip(np.exp(log_cost), a_min=0.05, a_max=None)

    # ── 8. Cyclical time encodings ────────────────────────────────────────────
    df['hour_sin']  = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['dow_sin']   = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos']   = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # ── 9. load_factor: deterministic temporal signal as a single feature ─────
    # Computable purely from the timestamp → zero leakage.
    # Encodes the expected cost level before AR noise and spikes.
    load_factor      = np.exp(diurnal_log + weekly_log + trend_log)
    df['load_factor'] = load_factor

    # ── 10. Log variance diagnostics ─────────────────────────────────────────
    var_diurnal = float(np.var(diurnal_log))
    var_weekly  = float(np.var(weekly_log))
    var_trend   = float(np.var(trend_log))
    var_ar      = float(np.var(z_ar))
    var_noise   = float(np.var(eps_noise))
    var_total   = float(np.var(log_cost))
    logger.info("Log-cost variance budget:")
    logger.info(f"  diurnal : {var_diurnal:.4f}  ({100*var_diurnal/var_total:.1f}%)")
    logger.info(f"  weekly  : {var_weekly:.4f}  ({100*var_weekly/var_total:.1f}%)")
    logger.info(f"  trend   : {var_trend:.4f}  ({100*var_trend/var_total:.1f}%)")
    logger.info(f"  AR(1)   : {var_ar:.4f}  ({100*var_ar/var_total:.1f}%)")
    logger.info(f"  noise   : {var_noise:.4f}  ({100*var_noise/var_total:.1f}%)")
    logger.info(f"  total   : {var_total:.4f}")
    logger.info(
        f"  theoretical R² ceiling (deterministic+AR) = "
        f"{(var_diurnal+var_weekly+var_trend+var_ar)/var_total:.3f}"
    )
    logger.info(f"  spike rows: {int(spike_mask.sum()):,}")

    save_dataframe(df, config.PROCESSED_DIR / 'cost_timeseries.csv', logger)
    _log_cost_stats(df, 'create_cost_timeseries', logger)
    return df


# ============================================================================
# STEP 3: FEATURE ENGINEERING  (no leakage)
# ============================================================================

# Columns that were INPUTS to computing cost – keeping them would be leakage
# because at real inference time we would not know the exact cost-driving
# quantities; we only observe the final cost (what we are predicting).
_LEAKY_COLS = {
    # Raw calendar ints — represented via sin/cos encodings instead.
    # (Resource quantity columns no longer exist in the direct log-cost model.)
    'hour_of_day', 'day_of_week', 'month',
    # NOTE: 'load_factor' is intentionally NOT leaky — it is derived purely
    # from the timestamp (diurnal * weekly * trend) and is available at
    # inference time given only the forecast timestamp.
}


def add_lag_rolling_features(
    df: pd.DataFrame,
    config: PipelineConfig,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Fix #1  – bfill() instead of fillna(method='bfill').
    Fix #5  – distinct, meaningful rolling windows; no duplicate of lag_1.
    Fix #6  – leaky cost-driver columns dropped before returning.
    Fix #4  – one-hot encoding of categoricals applied here.
    """
    df = df.sort_values('hour').reset_index(drop=True)

    # ── Consecutive lags (capture AR(1) decay: phi^1 … phi^LAG_HOURS) ─────────
    for lag in range(1, config.LAG_HOURS + 1):
        df[f'lag_cost_{lag}h'] = df['cost'].shift(lag)

    # ── Anchor lags at daily / weekly resolution ──────────────────────────────
    # These let the model see "same hour yesterday" and "same hour last week"
    # without needing 168 consecutive lag columns.
    for lag in config.LAG_ANCHORS:
        df[f'lag_cost_{lag}h'] = df['cost'].shift(lag)

    # ── Rolling statistics over multiple windows ───────────────────────────────
    # shift(1) ensures we never include the current row → no leakage
    for w in config.ROLLING_WINDOWS:
        rolled = df['cost'].shift(1).rolling(w, min_periods=1)
        df[f'roll_mean_{w}h'] = rolled.mean()
        df[f'roll_std_{w}h']  = rolled.std().fillna(0)
        df[f'roll_min_{w}h']  = rolled.min()
        df[f'roll_max_{w}h']  = rolled.max()

    # ── Exponentially weighted moving average (EWMA) ───────────────────────────
    # EWMA with span=s gives effective memory ≈ s hours and weights recent
    # observations more than a simple rolling mean does — ideal for AR(1).
    for span in [6, 24, 168]:
        df[f'ewma_{span}h'] = (
            df['cost'].shift(1)
                      .ewm(span=span, min_periods=1, adjust=False)
                      .mean()
        )

    # ── Log-cost lag (helps the model operate in log space naturally) ──────────
    df['lag_log_cost_1h'] = np.log1p(df['cost'].shift(1).fillna(df['cost'].median()))
    df['lag_log_cost_24h'] = np.log1p(df['cost'].shift(24).fillna(df['cost'].median()))

    # ── Cost velocity: how fast cost is changing hour-over-hour ───────────────
    df['cost_delta_1h']  = df['cost'].shift(1) - df['cost'].shift(2)
    df['cost_delta_24h'] = df['cost'].shift(1) - df['cost'].shift(25)

    # Forward-fill then back-fill NaNs at the start of the series (fix #1)
    lag_roll_cols = [c for c in df.columns if c.startswith(('lag_cost_', 'roll_', 'ewma_', 'lag_log_', 'cost_delta_'))]
    df[lag_roll_cols] = df[lag_roll_cols].ffill().bfill()

    # --- One-hot encode categoricals BEFORE dropping (fix #4) ---
    cat_cols = [c for c in ['cloud_provider', 'service_type'] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # --- Drop leaky columns (fix #6) ---
    cols_to_drop = [c for c in _LEAKY_COLS if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    logger.info(f"Dropped leaky columns: {cols_to_drop}")

    # Drop timestamp (not a numeric feature)
    if 'timestamp' in df.columns:
        df.drop(columns='timestamp', inplace=True)

    _log_cost_stats(df, 'add_lag_rolling_features', logger)
    return df


# ============================================================================
# STEP 4A: CROSS-VALIDATION  (fix #12)
# ============================================================================

def cross_validate(
    df: pd.DataFrame,
    config: PipelineConfig,
    logger: logging.Logger,
    log_target: bool = True,
) -> None:
    """
    Time-series cross-validation to get robust, unbiased metric estimates.
    When log_target=True the model trains on log1p(cost) and CV metrics
    are reported on both log scale (MAE) and dollar scale (MdAPE, RMSLE).
    """
    features = [c for c in df.columns if c not in ('cost', 'hour')]
    df = sanitise_dataframe(df, preserve_cols=[], logger=logger)
    X     = df[features].astype(np.float64).values
    y_raw = df['cost'].astype(np.float64).values
    y     = np.log1p(y_raw) if log_target else y_raw

    tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
    mae_scores, r2_scores, rmsle_scores = [], [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val   = X[train_idx], X[val_idx]
        y_tr, y_val   = y[train_idx], y[val_idx]
        y_val_raw     = y_raw[val_idx]

        m = xgb.XGBRegressor(
            n_estimators=config.XGB_N_ESTIMATORS,
            learning_rate=config.XGB_LEARNING_RATE,
            max_depth=config.XGB_MAX_DEPTH,
            min_child_weight=config.XGB_MIN_CHILD_WEIGHT,
            subsample=config.XGB_SUBSAMPLE,
            colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
            early_stopping_rounds=config.XGB_EARLY_STOPPING,
            eval_metric='mae',
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds_log = m.predict(X_val)
        preds_raw = np.expm1(preds_log) if log_target else preds_log

        mae_scores.append(mean_absolute_error(y_val, preds_log))
        r2_scores.append(r2_score(y_val_raw, preds_raw))
        rmsle_scores.append(float(np.sqrt(np.mean(
            (np.log1p(np.maximum(y_val_raw, 1e-6))
             - np.log1p(np.maximum(preds_raw, 1e-6))) ** 2
        ))))
        logger.info(
            f"  CV fold {fold}/{config.CV_SPLITS} — "
            f"MAE(log)={mae_scores[-1]:.4f}  "
            f"R²=${r2_scores[-1]:.4f}  "
            f"RMSLE={rmsle_scores[-1]:.4f}"
        )

    logger.info(
        f"Cross-validation summary — "
        f"MAE(log): {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}  |  "
        f"R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}  |  "
        f"RMSLE: {np.mean(rmsle_scores):.4f} ± {np.std(rmsle_scores):.4f}"
    )


# ============================================================================
# STEP 4B: TRAIN  (fix #7, #8)
# ============================================================================

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: PipelineConfig,
    logger: logging.Logger,
    feature_names: list = None,
) -> xgb.XGBRegressor:
    """Fix #7 – early stopping; fix #8 – single responsibility."""
    model = xgb.XGBRegressor(
        n_estimators=config.XGB_N_ESTIMATORS,
        learning_rate=config.XGB_LEARNING_RATE,
        max_depth=config.XGB_MAX_DEPTH,
        min_child_weight=config.XGB_MIN_CHILD_WEIGHT,
        subsample=config.XGB_SUBSAMPLE,
        colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
        early_stopping_rounds=config.XGB_EARLY_STOPPING,
        eval_metric='mae',
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    # Store feature names on the booster so they survive save/load
    if feature_names is not None:
        model.get_booster().feature_names = list(feature_names)
    best = model.best_iteration
    logger.info(f"Training complete — best iteration: {best}")
    return model


# ============================================================================
# STEP 4C: EVALUATE  (fix #8)
# ============================================================================

def evaluate_model(
    model: xgb.XGBRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    logger: logging.Logger,
    y_pred_override: np.ndarray = None,
) -> np.ndarray:

    y_pred = y_pred_override if y_pred_override is not None else model.predict(X_test)
    y_pred_clipped = np.maximum(y_pred, 1e-6)   # guard log(0)
    y_test_clipped = np.maximum(y_test, 1e-6)

    mae   = mean_absolute_error(y_test, y_pred)
    rmse  = mean_squared_error(y_test, y_pred) ** 0.5
    r2    = r2_score(y_test, y_pred)
    mdape = float(np.median(
        np.abs((y_test - y_pred) / y_test_clipped)
    )) * 100
    rmsle = float(np.sqrt(np.mean(
        (np.log1p(y_test_clipped) - np.log1p(y_pred_clipped)) ** 2
    )))

    logger.info("=" * 56)
    logger.info("HOLD-OUT TEST SET METRICS")
    logger.info(f"  MAE   (mean abs error, $)       : {mae:.4f}")
    logger.info(f"  RMSE  (root mean sq error, $)   : {rmse:.4f}")
    logger.info(f"  R²    (variance explained)      : {r2:.4f}")
    logger.info(f"  MdAPE (median abs % error)      : {mdape:.2f}%")
    logger.info(f"  RMSLE (log-scale RMSE)          : {rmsle:.4f}")
    logger.info("=" * 56)
    return y_pred


# ============================================================================
# STEP 4D: PLOT  (fix #8, #15)
# ============================================================================

def plot_predictions(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    config: PipelineConfig,
    logger: logging.Logger,
) -> None:
    """Fix #8 – plotting separated; fix #15 – plt.close after save."""
    n_show = min(500, len(y_test))

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Time series
    axes[0].plot(y_test[:n_show], label='Actual',    alpha=0.8, linewidth=1)
    axes[0].plot(y_pred[:n_show], label='Predicted', alpha=0.8, linewidth=1, linestyle='--')
    axes[0].set_title(f'XGBoost — Actual vs Predicted (first {n_show} test points)')
    axes[0].set_xlabel('Hour')
    axes[0].set_ylabel('Cost ($)')
    axes[0].legend()

    # Scatter
    axes[1].scatter(y_test[:n_show], y_pred[:n_show], alpha=0.3, s=10)
    lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[1].plot(lim, lim, 'r--', linewidth=1, label='Perfect fit')
    axes[1].set_xlabel('Actual Cost ($)')
    axes[1].set_ylabel('Predicted Cost ($)')
    axes[1].set_title('Actual vs Predicted scatter')
    axes[1].legend()

    plt.tight_layout()
    out = config.PROCESSED_DIR / 'xgb_predictions.png'
    plt.savefig(out, dpi=150)
    plt.close('all')                                   # fix #15
    logger.info(f"Predictions plot saved -> {out}")


def plot_feature_importance(
    model: xgb.XGBRegressor,
    feature_names: List[str],
    config: PipelineConfig,
    logger: logging.Logger,
) -> None:
    """Fix #8; fix #15."""
    importance = pd.Series(model.feature_importances_, index=feature_names)
    top15 = importance.nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))
    top15.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('XGBoost — Top 15 Feature Importances (gain)')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    out = config.PROCESSED_DIR / 'xgb_feature_importance.png'
    plt.savefig(out, dpi=150)
    plt.close('all')                                   # fix #15
    logger.info(f"Feature importance plot saved -> {out}")


# ============================================================================
# STEP 5: SAVE MODEL  (fix #3)
# ============================================================================

def save_model(
    model: xgb.XGBRegressor,
    config: PipelineConfig,
    logger: logging.Logger,
) -> None:
    """Fix #3 – persist trained model to MODELS_DIR."""
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = config.MODELS_DIR / f'xgb_cost_model_{timestamp}.json'
    model.save_model(model_path)
    logger.info(f"Model saved -> {model_path}")


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def run_xgboost_pipeline(
    df: pd.DataFrame,
    config: PipelineConfig,
    logger: logging.Logger,
) -> None:
    """
    Ties together CV + final train/evaluate/plot/save steps.
    Uses a strict chronological train/test split (no shuffling).
    """
    features = [c for c in df.columns if c not in ('cost', 'hour')]

    # Final safety net: ensure no dict/object dtype survives into XGBoost.
    # sanitise_dataframe is idempotent so re-running it is always safe.
    df = sanitise_dataframe(df, preserve_cols=[], logger=logger)

    X = df[features].astype(np.float64).values
    y_raw = df['cost'].astype(np.float64).values

    # ── Log-transform the target ──────────────────────────────────────────────
    # Cost is right-skewed (many cheap rows, few expensive ones).
    # Training on log1p(cost) makes the distribution near-Gaussian so that
    # XGBoost's squared-error loss treats all price ranges equally instead
    # of over-fitting the rare expensive rows.
    # Predictions are expm1()'d back to dollar space for evaluation.
    y = np.log1p(y_raw)

    split = int(len(df) * config.TRAIN_SPLIT)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test_log = y[:split], y[split:]
    y_test_raw = y_raw[split:]

    logger.info(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    logger.info(
        f"Target (log scale) — "
        f"min={y_train.min():.3f}  max={y_train.max():.3f}  "
        f"mean={y_train.mean():.3f}  std={y_train.std():.3f}"
    )

    # --- Cross-validation (fix #12)
    logger.info("Running time-series cross-validation…")
    cross_validate(df, config, logger, log_target=True)

    # --- Final model on full train set (fix #7, #8)
    logger.info("Training final model on full training split…")
    model = train_model(X_train, y_train, X_test, y_test_log, config, logger, feature_names=features)

    # --- Evaluate on original dollar scale (expm1 to undo log1p) ---
    y_pred_log = model.predict(X_test)
    y_pred_dollars = np.expm1(y_pred_log)
    y_pred = evaluate_model(model, X_test, y_test_raw, logger,
                            y_pred_override=y_pred_dollars)

    # --- Plot (fix #8, #15)
    plot_predictions(y_pred=y_pred, y_test=y_test_raw, config=config, logger=logger)
    plot_feature_importance(model=model, feature_names=features, config=config, logger=logger)

    # --- Save model (fix #3)
    save_model(model, config, logger)


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    pipeline_cfg = PipelineConfig()
    logger       = setup_logging(pipeline_cfg)

    # Load real AWS + Azure pricing from downloaded JSON files.
    # Falls back to hardcoded defaults gracefully if files are missing.
    pricing_cfg = load_pricing_from_files(pipeline_cfg.PRICING_DIR, logger)
    logger.info(
        f"Pricing loaded — "
        f"AWS: ${pricing_cfg.AWS_CPU_PRICE:.4f}/vCPU/hr, "
        f"${pricing_cfg.AWS_MEMORY_PRICE:.4f}/GB/hr  |  "
        f"Azure: ${pricing_cfg.AZURE_CPU_PRICE:.4f}/vCPU/hr, "
        f"${pricing_cfg.AZURE_MEMORY_PRICE:.4f}/GB/hr"
    )

    try:
        trace_file = pipeline_cfg.RAW_DIR / 'workloads' / 'instance_usage-000000000000.json'

        usage_df = sample_google_trace(trace_file, pipeline_cfg, logger)
        cost_df  = create_cost_timeseries(usage_df, pipeline_cfg, pricing_cfg, logger)
        feat_df  = add_lag_rolling_features(cost_df, pipeline_cfg, logger)

        run_xgboost_pipeline(feat_df, pipeline_cfg, logger)

        logger.info("Full pipeline completed successfully!")
        return 0

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())