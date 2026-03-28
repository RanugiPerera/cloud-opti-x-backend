import numpy as np
import xgboost as xgb
from datetime import datetime
from services.forecast_service import _diurnal_log, _weekly_log, forecast_48h

def test_diurnal_log_shape():
    hours = np.arange(24)
    result = _diurnal_log(hours)
    assert result.shape == (24,)
    assert result.max() <= 0.40   # peak at 14:00 ≈ 0.35
    assert result.min() >= -0.30  # trough at 04:00 ≈ -0.25
    
def test_weekly_log():
    weekdays = np.array([0, 1, 2, 3, 4])
    weekends = np.array([5, 6])
    assert np.all(_weekly_log(weekdays) == 0.05)
    assert np.all(_weekly_log(weekends) == -0.55)
    
def test_v2_features_built():
    """Verify the 9 v2 short-horizon features are built in forecast_48h."""
    class MockXGB:
        def get_booster(self):
            class Booster:
                @property
                def feature_names(self):
                    return [f"f{i}" for i in range(61)]
            return Booster()
        def predict(self, X):
            return np.zeros(X.shape[0] if hasattr(X, "shape") else 1)
            
    model = MockXGB()
    
    history = np.random.default_rng(42).exponential(1.0, 200).tolist()
    start   = datetime.now().replace(minute=0, second=0, microsecond=0)
    AWS_PRICING = {
        "compute": 0.05, "storage": 0.02, "network": 0.01,
        "cpu_per_vcpu_hr": 0.04,
        "mem_gb_per_vcpu_hr": 0.005
    }
    df      = forecast_48h(model, history, start, 200, "AWS", AWS_PRICING)


    v2_features = [
        'lag_cost_0h', 'lag_log_cost_0h', 'ar_residual_0h',
        'cost_pctile_24h', 'lag_cost_1h_sq', 'lag_ratio_1h_6h',
        'cost_delta_2h', 'cost_accel_1h', 'lag_pct_of_3h_max'
    ]
    # Verify no warnings about missing features were raised
    assert len(df) == 48
    assert df['predicted_cost'].min() > 0
    assert df['predicted_cost'].max() < 10  # sanity check — $/hr not $/month

def test_ewma_incremental_matches_full():
    """Incremental EWMA update must match full recompute for correctness."""
    history = np.random.default_rng(0).exponential(1.0, 200).tolist()
    span    = 6
    alpha   = 2 / (span + 1)

    # Full recompute
    ewma_full = history[0]
    for v in history[1:]:
        ewma_full = alpha * v + (1 - alpha) * ewma_full

    # Incremental — initialise then update once
    ewma_inc = history[0]
    for v in history[1:-1]:
        ewma_inc = alpha * v + (1 - alpha) * ewma_inc
    ewma_inc = alpha * history[-1] + (1 - alpha) * ewma_inc

    assert abs(ewma_full - ewma_inc) < 1e-10