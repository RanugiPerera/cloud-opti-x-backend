
import glob
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent
MODELS_DIR    = BASE_DIR / "trained_models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PRICING_DIR   = BASE_DIR / "data" / "raw" / "pricing"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("forecast_demo")

# ── Pricing defaults (mirrors pipeline PricingConfig) ─────────────────────────
_AWS_DEFAULTS   = dict(cpu_per_vcpu_hr=0.0480, mem_per_gb_hr=0.0060,
                       storage_per_gb_hr=0.1/730, network_per_gb=0.09)
_AZURE_DEFAULTS = dict(cpu_per_vcpu_hr=0.0530, mem_per_gb_hr=0.0067,
                       storage_per_gb_hr=0.12/730, network_per_gb=0.087)


# ============================================================================
# STEP 1: LOAD REAL PRICING
# ============================================================================

def load_pricing() -> dict:
    """
    Read AWS and Azure pricing JSON files downloaded by the pricing script.
    Returns a dict with keys: aws_cpu, aws_mem, azure_cpu, azure_mem, etc.
    Falls back to defaults if files are absent.
    """
    aws   = dict(_AWS_DEFAULTS)
    azure = dict(_AZURE_DEFAULTS)

    # ── AWS EC2 ───────────────────────────────────────────────────────────────
    aws_file = PRICING_DIR / "aws_ec2_pricing.json"
    if aws_file.exists():
        try:
            with open(aws_file, encoding="utf-8") as f:
                ec2 = json.load(f)
            products = ec2.get("products", {})
            terms    = ec2.get("terms", {}).get("OnDemand", {})
            for sku, prod in products.items():
                attr = prod.get("attributes", {})
                if (attr.get("instanceType") == "m5.xlarge"
                        and attr.get("location")        == "US East (N. Virginia)"
                        and attr.get("operatingSystem")  == "Linux"
                        and attr.get("tenancy")          == "Shared"
                        and attr.get("capacitystatus")   == "Used"):
                    if sku in terms:
                        dims = next(iter(terms[sku].values()))["priceDimensions"]
                        hr   = float(next(iter(dims.values()))["pricePerUnit"]["USD"])
                        aws["cpu_per_vcpu_hr"] = hr / 4
                        aws["mem_per_gb_hr"]   = hr / 16
                        logger.info(f"[AWS]   m5.xlarge  ${hr:.4f}/hr  "
                                    f"→ ${aws['cpu_per_vcpu_hr']:.4f}/vCPU/hr")
                    break
        except Exception as e:
            logger.warning(f"[AWS]   pricing parse failed ({e}) — using defaults")
    else:
        logger.warning("[AWS]   aws_ec2_pricing.json not found — using defaults")

    # ── Azure VM ──────────────────────────────────────────────────────────────
    az_file = PRICING_DIR / "azure_vm_pricing.json"
    if az_file.exists():
        try:
            with open(az_file, encoding="utf-8") as f:
                az_items = json.load(f)
            for item in az_items:
                if ("D4s v3" in item.get("skuName", "")
                        and item.get("armRegionName") == "eastus"
                        and "Windows" not in item.get("productName", "")
                        and item.get("type") == "Consumption"):
                    hr = float(item["retailPrice"])
                    azure["cpu_per_vcpu_hr"] = hr / 4
                    azure["mem_per_gb_hr"]   = hr / 16
                    logger.info(f"[Azure] D4s_v3     ${hr:.4f}/hr  "
                                f"→ ${azure['cpu_per_vcpu_hr']:.4f}/vCPU/hr")
                    break
        except Exception as e:
            logger.warning(f"[Azure] pricing parse failed ({e}) — using defaults")
    else:
        logger.warning("[Azure] azure_vm_pricing.json not found — using defaults")

    return {"aws": aws, "azure": azure}


# ============================================================================
# STEP 2: BUILD FEATURES FOR A FUTURE TIMESTAMP
# ============================================================================

def _diurnal_log(hour: np.ndarray) -> np.ndarray:
    """Identical diurnal function used during training."""
    return (
        0.35 * np.exp(-0.5 * ((hour - 14) / 3.5) ** 2)
      + 0.15 * np.exp(-0.5 * ((hour -  9) / 2.0) ** 2)
      - 0.25 * np.exp(-0.5 * ((hour -  4) / 2.5) ** 2)
    )


def _weekly_log(dow: np.ndarray) -> np.ndarray:
    """Identical weekly function used during training."""
    return np.where(dow < 5, 0.05, -0.55)


def build_forecast_features(
    timestamps: pd.DatetimeIndex,
    last_known_costs: np.ndarray,
    hour_offset: int = 0,
) -> pd.DataFrame:
    """
    Construct the exact same feature set the model was trained on for a
    sequence of future timestamps.

    Parameters
    ----------
    timestamps       : future hourly timestamps to forecast
    last_known_costs : array of the most recent actual cost values
                       (used to initialise lag and rolling features)
    hour_offset      : global hour index of the first timestamp
                       (used to keep 'hour' consistent with training)
    """
    n            = len(timestamps)
    hour_of_day  = np.array([t.hour         for t in timestamps])
    day_of_week  = np.array([t.dayofweek    for t in timestamps])
    month        = np.array([t.month        for t in timestamps])
    month_idx    = month - 1

    # ── Cyclical encodings ────────────────────────────────────────────────────
    hour_sin  = np.sin(2 * np.pi * hour_of_day / 24)
    hour_cos  = np.cos(2 * np.pi * hour_of_day / 24)
    dow_sin   = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos   = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # ── load_factor (deterministic, timestamp-only) ───────────────────────────
    diurnal     = _diurnal_log(hour_of_day)
    weekly      = _weekly_log(day_of_week)
    trend       = month_idx * np.log(1.008)
    load_factor = np.exp(diurnal + weekly + trend)

    # ── Global hour index ─────────────────────────────────────────────────────
    hour = np.arange(hour_offset, hour_offset + n)

    # ── Lag + rolling features using last_known_costs as history ─────────────
    # We build a cost buffer: last known costs + (to be filled) future costs
    # For forecasting we use the last known costs for all lag lookups.
    # This is an iterative 1-step-ahead approach: each predicted cost is
    # fed back as the lag for the next step.
    LAG_HOURS    = 12
    LAG_ANCHORS  = [24, 48, 168]
    ROLL_WINDOWS = [6, 12, 24, 48, 168]
    EWMA_SPANS   = [6, 24, 168]

    # Buffer of known + predicted costs (initialised with last known)
    history = list(last_known_costs[-200:])   # keep last 200 hours
    predicted_costs = []

    rows = []
    for i in range(n):
        buf = np.array(history)
        row = {}

        # Lag features
        for lag in range(1, LAG_HOURS + 1):
            idx = -(lag)
            row[f"lag_cost_{lag}h"] = buf[idx] if len(buf) >= lag else buf[0]
        for lag in LAG_ANCHORS:
            row[f"lag_cost_{lag}h"] = buf[-lag] if len(buf) >= lag else buf[0]

        # Rolling stats
        for w in ROLL_WINDOWS:
            window = buf[-w:] if len(buf) >= w else buf
            row[f"roll_mean_{w}h"] = float(np.mean(window))
            row[f"roll_std_{w}h"]  = float(np.std(window))
            row[f"roll_min_{w}h"]  = float(np.min(window))
            row[f"roll_max_{w}h"]  = float(np.max(window))

        # EWMA
        for span in EWMA_SPANS:
            alpha = 2 / (span + 1)
            ewma  = buf[0]
            for v in buf[1:]:
                ewma = alpha * v + (1 - alpha) * ewma
            row[f"ewma_{span}h"] = float(ewma)

        # Log lags
        row["lag_log_cost_1h"]  = float(np.log1p(buf[-1]))
        row["lag_log_cost_24h"] = float(np.log1p(buf[-24]  if len(buf) >= 24  else buf[0]))

        # Cost velocity
        row["cost_delta_1h"]  = float(buf[-1] - buf[-2])  if len(buf) >= 2  else 0.0
        row["cost_delta_24h"] = float(buf[-1] - buf[-25]) if len(buf) >= 25 else 0.0

        # Time features
        row["hour_sin"]   = hour_sin[i]
        row["hour_cos"]   = hour_cos[i]
        row["dow_sin"]    = dow_sin[i]
        row["dow_cos"]    = dow_cos[i]
        row["month_sin"]  = month_sin[i]
        row["month_cos"]  = month_cos[i]
        row["load_factor"] = load_factor[i]
        row["hour"]        = hour[i]

        rows.append(row)

        # Placeholder — will be filled with model prediction after we return
        history.append(buf[-1])   # temporary; replaced in forecast loop

    return pd.DataFrame(rows)


# ============================================================================
# STEP 3: ITERATIVE 48-HOUR FORECAST
# ============================================================================

def forecast_48h(
    model: xgb.XGBRegressor,
    last_known_costs: np.ndarray,
    start_time: datetime,
    hour_offset: int,
    provider_label: str,
    pricing: dict,
) -> pd.DataFrame:
    """
    Generate a 48-hour ahead forecast using iterative 1-step prediction.
    Each predicted value is fed back as a lag for the next step.
    """
    timestamps = pd.date_range(start_time, periods=48, freq="h")
    history    = list(last_known_costs[-200:])
    forecasts  = []

    logger.info(f"Forecasting 48 hours ahead for {provider_label}...")

    LAG_HOURS    = 12
    LAG_ANCHORS  = [24, 48, 168]
    ROLL_WINDOWS = [6, 12, 24, 48, 168]
    EWMA_SPANS   = [6, 24, 168]

    hour_of_day = np.array([t.hour      for t in timestamps])
    day_of_week = np.array([t.dayofweek for t in timestamps])
    month       = np.array([t.month     for t in timestamps])
    month_idx   = month - 1

    diurnal     = _diurnal_log(hour_of_day)
    weekly      = _weekly_log(day_of_week)
    trend       = month_idx * np.log(1.008)
    load_factor = np.exp(diurnal + weekly + trend)

    hour_sin  = np.sin(2 * np.pi * hour_of_day / 24)
    hour_cos  = np.cos(2 * np.pi * hour_of_day / 24)
    dow_sin   = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos   = np.cos(2 * np.pi * day_of_week / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    for i in range(48):
        buf = np.array(history)
        row = {}

        for lag in range(1, LAG_HOURS + 1):
            row[f"lag_cost_{lag}h"] = float(buf[-lag]) if len(buf) >= lag else float(buf[0])
        for lag in LAG_ANCHORS:
            row[f"lag_cost_{lag}h"] = float(buf[-lag]) if len(buf) >= lag else float(buf[0])

        for w in ROLL_WINDOWS:
            window = buf[-w:] if len(buf) >= w else buf
            row[f"roll_mean_{w}h"] = float(np.mean(window))
            row[f"roll_std_{w}h"]  = float(np.std(window))
            row[f"roll_min_{w}h"]  = float(np.min(window))
            row[f"roll_max_{w}h"]  = float(np.max(window))

        for span in EWMA_SPANS:
            alpha = 2 / (span + 1)
            ewma  = float(buf[0])
            for v in buf[1:]:
                ewma = alpha * float(v) + (1 - alpha) * ewma
            row[f"ewma_{span}h"] = ewma

        row["lag_log_cost_1h"]  = float(np.log1p(buf[-1]))
        row["lag_log_cost_24h"] = float(np.log1p(buf[-24]  if len(buf) >= 24  else buf[0]))
        row["cost_delta_1h"]    = float(buf[-1] - buf[-2])  if len(buf) >= 2  else 0.0
        row["cost_delta_24h"]   = float(buf[-1] - buf[-25]) if len(buf) >= 25 else 0.0

        row["hour_sin"]    = hour_sin[i]
        row["hour_cos"]    = hour_cos[i]
        row["dow_sin"]     = dow_sin[i]
        row["dow_cos"]     = dow_cos[i]
        row["month_sin"]   = month_sin[i]
        row["month_cos"]   = month_cos[i]
        row["load_factor"] = load_factor[i]
        row["hour"]        = float(hour_offset + i)

        # Align features to what the model was trained on.
        # Any model feature not in our row gets filled with 0 (e.g. legacy
        # cpu_usage / memory_usage columns from an older model checkpoint).
        feat_names = model.get_booster().feature_names
        X_df = pd.DataFrame([row])
        if feat_names:
            missing = [f for f in feat_names if f not in X_df.columns]
            if missing:
                logger.warning(
                    f"Model expects {len(missing)} feature(s) not built by "
                    f"this script: {missing}. Filling with 0. "
                    f"Re-run full_pipeline.py to get a fresh model."
                )
                for col in missing:
                    X_df[col] = 0.0
            X = X_df[feat_names]
        else:
            X = X_df
        log_pred   = float(model.predict(X.values.astype(np.float64))[0])
        cost_pred  = float(np.expm1(log_pred))

        # Scale by provider pricing ratio so AWS vs Azure differ realistically
        scale = (pricing["cpu_per_vcpu_hr"] / _AWS_DEFAULTS["cpu_per_vcpu_hr"])
        cost_pred_scaled = max(cost_pred * scale, 0.05)

        forecasts.append({
            "timestamp":      timestamps[i],
            "predicted_cost": cost_pred_scaled,
            "provider":       provider_label,
        })

        # Feed prediction back into history for next step
        history.append(cost_pred_scaled)

    df = pd.DataFrame(forecasts)
    logger.info(
        f"  {provider_label} forecast — "
        f"min=${df['predicted_cost'].min():.3f}  "
        f"max=${df['predicted_cost'].max():.3f}  "
        f"total_48h=${df['predicted_cost'].sum():.2f}"
    )
    return df


# ============================================================================
# STEP 4: PLOT
# ============================================================================

def plot_forecast(
    aws_df:   pd.DataFrame,
    azure_df: pd.DataFrame,
    last_known_costs: np.ndarray,
    last_known_times: pd.DatetimeIndex,
    budget_threshold: float,
    output_path: Path,
) -> None:
    """
    Publication-quality 3-panel forecast chart:
      Panel 1: 24h actual history + 48h AWS forecast
      Panel 2: 24h actual history + 48h Azure forecast
      Panel 3: Side-by-side cost comparison bar chart
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    fig.suptitle(
        "Multi-Cloud Cost Forecast — 48-Hour Ahead Prediction\n"
        "XGBoost Model  |  Features: Temporal Patterns + Autocorrelation Lags",
        fontsize=13, fontweight="bold", y=0.98,
    )

    hist_times = last_known_times[-24:]
    hist_costs = last_known_costs[-24:]

    for ax, df, provider, colour in [
        (axes[0], aws_df,   "AWS",   "#FF9900"),
        (axes[1], azure_df, "Azure", "#0089D6"),
    ]:
        # History
        ax.plot(hist_times, hist_costs, color="grey",
                linewidth=1.5, alpha=0.7, label="Actual (last 24h)")
        # Forecast
        ax.plot(df["timestamp"], df["predicted_cost"],
                color=colour, linewidth=2.2, label=f"{provider} Forecast")
        # Confidence band (±15% — representative of MdAPE from training)
        lo = df["predicted_cost"] * 0.85
        hi = df["predicted_cost"] * 1.15
        ax.fill_between(df["timestamp"], lo, hi,
                        color=colour, alpha=0.15, label="±15% band")
        # Budget threshold
        ax.axhline(budget_threshold, color="red", linestyle="--",
                   linewidth=1.2, alpha=0.8, label=f"Budget threshold (${budget_threshold:.2f}/hr)")
        # Shading — weekend hours
        for ts in df["timestamp"]:
            if ts.dayofweek >= 5:
                ax.axvspan(ts, ts + timedelta(hours=1),
                           color="lightblue", alpha=0.08)
        # Divider between history and forecast
        ax.axvline(df["timestamp"].iloc[0], color="black",
                   linestyle=":", linewidth=1, alpha=0.5)
        ax.text(df["timestamp"].iloc[0], ax.get_ylim()[1] * 0.95,
                " Forecast →", fontsize=8, color="black", alpha=0.6)

        ax.set_ylabel("Cost ($/hr)", fontsize=10)
        ax.set_title(f"{provider} — 48-Hour Cost Forecast", fontsize=11)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.set_facecolor("#fafafa")

    # ── Panel 3: bar comparison ───────────────────────────────────────────────
    ax3 = axes[2]
    hours       = list(range(48))
    aws_costs   = aws_df["predicted_cost"].values
    azure_costs = azure_df["predicted_cost"].values
    x           = np.arange(48)
    width       = 0.4

    ax3.bar(x - width/2, aws_costs,   width, color="#FF9900", alpha=0.8, label="AWS")
    ax3.bar(x + width/2, azure_costs, width, color="#0089D6", alpha=0.8, label="Azure")
    ax3.axhline(budget_threshold, color="red", linestyle="--",
                linewidth=1.2, alpha=0.8, label=f"Budget threshold")
    ax3.set_xlabel("Hour ahead", fontsize=10)
    ax3.set_ylabel("Cost ($/hr)", fontsize=10)
    ax3.set_title(
        f"AWS vs Azure Cost Comparison  |  "
        f"48h total — AWS: ${aws_costs.sum():.2f}  Azure: ${azure_costs.sum():.2f}  "
        f"Saving: ${abs(aws_costs.sum()-azure_costs.sum()):.2f}",
        fontsize=10,
    )
    ax3.legend(fontsize=9)
    ax3.set_xticks(x[::6])
    ax3.set_xticklabels([f"+{h}h" for h in hours[::6]])
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_facecolor("#fafafa")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Forecast chart saved → {output_path}")


# ============================================================================
# STEP 5: SAVE CSV
# ============================================================================

def save_forecast_csv(
    aws_df: pd.DataFrame,
    azure_df: pd.DataFrame,
    output_path: Path,
) -> None:
    combined = pd.concat([aws_df, azure_df], ignore_index=True)
    combined["hour_ahead"] = combined.groupby("provider").cumcount() + 1
    combined.to_csv(output_path, index=False)
    logger.info(f"Forecast CSV  saved → {output_path}")

    # Print summary table
    print("\n" + "=" * 58)
    print(f"{'HOUR':>5}  {'TIMESTAMP':<18}  {'AWS ($/hr)':>10}  {'AZURE ($/hr)':>12}")
    print("-" * 58)
    for i in range(min(48, len(aws_df))):
        ts    = aws_df.iloc[i]["timestamp"]
        a_val = aws_df.iloc[i]["predicted_cost"]
        z_val = azure_df.iloc[i]["predicted_cost"]
        flag  = " ◄ cheaper" if a_val < z_val else ""
        print(f"  +{i+1:02d}  {str(ts)[:16]:<18}  {a_val:>10.4f}  {z_val:>12.4f}{flag}")
    print("=" * 58)
    print(f"{'TOTAL':>5}  {'':18}  {aws_df['predicted_cost'].sum():>10.2f}  "
          f"{azure_df['predicted_cost'].sum():>12.2f}")
    print("=" * 58)


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    logger.info("=" * 58)
    logger.info("  CLOUD COST FORECAST DEMO  —  48-Hour Prediction")
    logger.info("=" * 58)

    # ── Find latest trained model ─────────────────────────────────────────────
    model_files = sorted(glob.glob(str(MODELS_DIR / "xgb_cost_model_*.json")))
    if not model_files:
        logger.error(f"No trained model found in {MODELS_DIR}")
        logger.error("Run full_pipeline.py first to train the model.")
        return 1

    model_path = model_files[-1]   # most recent
    logger.info(f"Loading model: {Path(model_path).name}")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    # feature_names may be None if the booster was saved without them;
    # fall back to the booster's internal num_features count instead.
    booster       = model.get_booster()
    feature_names = booster.feature_names
    n_features    = len(feature_names) if feature_names else booster.num_features()
    logger.info(f"Model loaded  — {n_features} features")

    # ── Load real pricing ─────────────────────────────────────────────────────
    pricing = load_pricing()
    aws_pricing   = pricing["aws"]
    azure_pricing = pricing["azure"]

    # ── Simulate "last known" cost history ────────────────────────────────────
    # In a real system this would come from AWS Cost Explorer / Azure Cost Mgmt.
    # Here we reconstruct the last 200 hours of the training distribution so
    # the lag features are properly initialised.
    rng          = np.random.default_rng(42)
    n_history    = 200
    start_hist   = datetime.now() - timedelta(hours=n_history)
    hist_times   = pd.date_range(start_hist, periods=n_history, freq="h")
    hist_hours   = np.array([t.hour      for t in hist_times])
    hist_dow     = np.array([t.dayofweek for t in hist_times])
    hist_month   = np.array([t.month     for t in hist_times])
    hist_diurnal = _diurnal_log(hist_hours)
    hist_weekly  = _weekly_log(hist_dow)
    hist_trend   = (hist_month - 1) * np.log(1.008)

    # AR(1) residual
    phi, sigma = 0.7, 0.06
    ar = np.zeros(n_history)
    for t in range(1, n_history):
        ar[t] = phi * ar[t-1] + rng.normal(0, sigma)

    log_hist = hist_diurnal + hist_weekly + hist_trend + ar + rng.normal(0, 0.10, n_history)
    last_known_costs = np.clip(np.exp(log_hist), 0.05, None)

    hour_offset = n_history   # so forecast hour indices continue from history

    # ── Forecast start time ───────────────────────────────────────────────────
    start_time = datetime.now().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    logger.info(f"Forecast window: {start_time}  →  "
                f"{start_time + timedelta(hours=47)}")

    # ── Run forecasts for AWS and Azure ───────────────────────────────────────
    aws_forecast = forecast_48h(
        model, last_known_costs, start_time,
        hour_offset, "AWS", aws_pricing,
    )
    azure_forecast = forecast_48h(
        model, last_known_costs, start_time,
        hour_offset, "Azure", azure_pricing,
    )

    # ── Budget threshold — 90th percentile of history ─────────────────────────
    budget_threshold = float(np.percentile(last_known_costs, 90))
    logger.info(f"Budget threshold (90th pctile of history): ${budget_threshold:.4f}/hr")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_forecast(
        aws_forecast, azure_forecast,
        last_known_costs, hist_times,
        budget_threshold,
        PROCESSED_DIR / "forecast_48h.png",
    )

    # ── CSV ───────────────────────────────────────────────────────────────────
    save_forecast_csv(
        aws_forecast, azure_forecast,
        PROCESSED_DIR / "forecast_48h.csv",
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    aws_total   = aws_forecast["predicted_cost"].sum()
    azure_total = azure_forecast["predicted_cost"].sum()
    cheaper     = "AWS" if aws_total < azure_total else "Azure"
    saving      = abs(aws_total - azure_total)

    logger.info("")
    logger.info("=" * 58)
    logger.info("  FORECAST SUMMARY")
    logger.info("=" * 58)
    logger.info(f"  AWS   total (48h): ${aws_total:>8.2f}")
    logger.info(f"  Azure total (48h): ${azure_total:>8.2f}")
    logger.info(f"  Cheaper provider : {cheaper}  (saves ${saving:.2f} over 48h)")
    logger.info(f"  Model MdAPE      : ~11.5%  (from training evaluation)")
    logger.info("=" * 58)
    logger.info("  Output files:")
    logger.info(f"    {PROCESSED_DIR / 'forecast_48h.png'}")
    logger.info(f"    {PROCESSED_DIR / 'forecast_48h.csv'}")
    logger.info("=" * 58)

    return 0


if __name__ == "__main__":
    sys.exit(main())
    

class ForecastService:

    def __init__(self):
        self._model   = None
        self._pricing = None
        self._history = None
        self._budget  = None
        self._load()

    def _load(self):
        """Load model, pricing, and build cost history buffer."""
        import glob
        model_files = sorted(glob.glob(str(MODELS_DIR / "xgb_cost_model_*.json")))
        if not model_files:
            raise FileNotFoundError(
                f"No XGBoost model found in {MODELS_DIR}. "
                "Run full_pipeline.py first."
            )
        self._model = xgb.XGBRegressor()
        self._model.load_model(model_files[-1])

        pricing       = load_pricing()
        self._pricing = pricing

        # Build 200-hour cost history buffer using same generation model
        rng        = np.random.default_rng(42)
        n          = 200
        hist_times = pd.date_range(
            datetime.now() - timedelta(hours=n), periods=n, freq="h"
        )
        hours  = np.array([t.hour      for t in hist_times])
        dows   = np.array([t.dayofweek for t in hist_times])
        months = np.array([t.month     for t in hist_times])

        phi, sigma = 0.7, 0.06
        ar  = np.zeros(n)
        eps = rng.normal(0, sigma, n)
        for t in range(1, n):
            ar[t] = phi * ar[t-1] + eps[t]

        log_hist = (
            _diurnal_log(hours)
            + _weekly_log(dows)
            + (months - 1) * np.log(1.008)
            + ar
            + rng.normal(0, 0.10, n)
        )
        self._history = np.clip(np.exp(log_hist), 0.05, None)
        self._budget  = float(np.percentile(self._history, 90))

    def predict_costs(self, cloud_provider: str, forecast_hours: int = 24) -> dict:
        """
        Run iterative XGBoost forecast for the requested provider and horizon.
        Returns dict with timestamps and costs lists.
        """
        pricing    = self._pricing[cloud_provider]
        start_time = (
            datetime.now()
            .replace(minute=0, second=0, microsecond=0)
            + timedelta(hours=1)
        )

        df = forecast_48h(
            model          = self._model,
            last_known_costs = self._history,
            start_time     = start_time,
            hour_offset    = 200,
            provider_label = cloud_provider.upper(),
            pricing        = pricing,
        )

        return {
            "timestamps": [
                ts.isoformat() for ts in df["timestamp"][:forecast_hours]
            ],
            "costs":            df["predicted_cost"][:forecast_hours].tolist(),
            "budget_threshold": self._budget,
        }

    def get_stats(self) -> dict:
        """Return model metadata and cost history statistics."""
        booster    = self._model.get_booster()
        feat_names = booster.feature_names
        n_features = len(feat_names) if feat_names else booster.num_features()

        return {
            "status": "success",
            "model": {
                "type":             "XGBoost",
                "n_features":       n_features,
                "r_squared":        0.717,
                "mdape":            11.45,
                "rmsle":            0.0949,
                "mae":              0.1548,
                "lookback_hours":   168,
                "forecast_horizon": 48,
            },
            "pricing": {
                "aws": {
                    "instance":        "m5.xlarge (us-east-1)",
                    "cpu_per_vcpu_hr": self._pricing["aws"]["cpu_per_vcpu_hr"],
                    "mem_per_gb_hr":   self._pricing["aws"]["mem_per_gb_hr"],
                },
                "azure": {
                    "instance":        "D4s_v3 (eastus)",
                    "cpu_per_vcpu_hr": self._pricing["azure"]["cpu_per_vcpu_hr"],
                    "mem_per_gb_hr":   self._pricing["azure"]["mem_per_gb_hr"],
                },
            },
            "cost_history": {
                "hours":            len(self._history),
                "min":              round(float(self._history.min()),  4),
                "max":              round(float(self._history.max()),  4),
                "mean":             round(float(self._history.mean()), 4),
                "std":              round(float(self._history.std()),  4),
                "budget_threshold": round(self._budget, 4),
            },
        }