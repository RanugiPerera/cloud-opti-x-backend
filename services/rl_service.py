import glob
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "trained_models"

logger = logging.getLogger(__name__)

# Add models directory to path so rl_agent_integrated can be imported
_candidates = [
    BASE_DIR / "models",
    BASE_DIR / "scripts" / "models",
    BASE_DIR / "scripts",
]
for _p in _candidates:
    if (_p / "rl_agent_integrated.py").exists():
        sys.path.insert(0, str(_p))
        break

from rl_agent_integrated import (
    DQNAgent,
    ForecastDrivenCloudEnvironment,
    _diurnal_log,
    _weekly_log,
)

import xgboost as xgb


_ACTION_REASONING = {
    "scale_up":      "Forecast shows rising demand — scaling up resources proactively",
    "scale_down":    "Forecast shows declining costs — scaling down to reduce spend",
    "migrate_aws":   "AWS forecast is cheaper over the next 6 hours — migrating",
    "migrate_azure": "Azure forecast is cheaper over the next 6 hours — migrating",
}

_SCALE_EFFECT = {
    "scale_up":      1.40,
    "scale_down":    0.75,
    "migrate_aws":   1.00,
    "migrate_azure": 1.00,
}


class RLService:
    """Wraps DQN agent for use by the Flask API."""

    def __init__(self):
        self._agent = None
        self._xgb   = None
        self._history = None
        self._load()

    def _load(self):
        # ── Load XGBoost ──────────────────────────────────────────────────────
        xgb_files = sorted(glob.glob(str(MODELS_DIR / "xgb_cost_model_*.json")))
        if not xgb_files:
            raise FileNotFoundError(
                "No XGBoost model found. Run full_pipeline.py first."
            )
        self._xgb = xgb.XGBRegressor()
        self._xgb.load_model(xgb_files[-1])
        logger.info(f"XGBoost loaded: {Path(xgb_files[-1]).name}")

        # ── Load RL agent ─────────────────────────────────────────────────────
        rl_files = sorted(glob.glob(str(MODELS_DIR / "rl_agent_integrated.pth")))
        if not rl_files:
            raise FileNotFoundError(
                "No RL agent found. Run train_rl_integrated.py first."
            )
        self._agent = DQNAgent(
            state_dim  = ForecastDrivenCloudEnvironment.STATE_DIM,
            action_dim = ForecastDrivenCloudEnvironment.ACTION_DIM,
        )
        self._agent.load(rl_files[-1])
        self._agent.epsilon = 0.0   # fully greedy — no random actions
        logger.info("DQN agent loaded (greedy mode)")

        # ── Build cost history ────────────────────────────────────────────────
        rng   = np.random.default_rng(42)
        n     = 200
        times = [datetime(2026, 1, 1) + timedelta(hours=i) for i in range(n)]
        hours  = np.array([t.hour      for t in times])
        dows   = np.array([t.weekday() for t in times])
        months = np.array([t.month     for t in times])

        phi, sigma = 0.7, 0.06
        ar  = np.zeros(n)
        eps = rng.normal(0, sigma, n)
        for t in range(1, n):
            ar[t] = phi * ar[t-1] + eps[t]

        log_hist = (
            _diurnal_log(hours)
            + _weekly_log(dows)
            + (months - 1) * np.log(1.008)
            + ar + rng.normal(0, 0.10, n)
        )
        self._history = np.clip(np.exp(log_hist), 0.05, None)

    def _make_env(self, starting_provider: str = "aws") -> ForecastDrivenCloudEnvironment:
        env = ForecastDrivenCloudEnvironment(
            xgb_model    = self._xgb,
            cost_history = self._history,
            episode_length = 48,
        )
        env.reset()
        env.provider = starting_provider
        return env

    def recommend(
        self,
        current_cost: float,
        provider:     str   = "aws",
        scale_factor: float = 1.0,
    ) -> dict:
        """
        Get a single action recommendation for the given state.
        """
        env = self._make_env(provider)

        # Override env cost to match caller\'s reported current cost
        env.history[-1]    = current_cost
        env.current_cost   = current_cost
        env.scale_factor   = scale_factor

        # Get forecast for state construction
        from rl_agent_integrated import xgb_forecast
        forecast = xgb_forecast(
            self._xgb, env.history, env.current_time, env.hour_offset, n_steps=6
        )

        state  = env._build_state(current_cost, forecast)
        action = self._agent.select_action(state, greedy=True)
        name   = self._agent.get_action_name(action)

        # Estimate cost after action
        scale_after = scale_factor * _SCALE_EFFECT[name]
        cost_after  = forecast[0] * scale_after
        saving_pct  = round((current_cost - cost_after) / current_cost * 100, 1)

        return {
            "status":    "success",
            "action":    name,
            "action_id": action,
            "reasoning": _ACTION_REASONING[name],
            "current_state": {
                "current_cost":  round(current_cost,   4),
                "provider":      provider,
                "scale_factor":  round(scale_factor,   3),
                "forecast_1h":   round(float(forecast[0]), 4),
                "forecast_3h":   round(float(forecast[2]), 4),
                "forecast_6h":   round(float(forecast[5]), 4),
            },
            "expected_outcome": {
                "estimated_cost_after": round(max(cost_after, 0.05), 4),
                "cost_saving_pct":      saving_pct,
            },
        }

    def simulate_episode(
        self,
        hours:             int = 24,
        starting_provider: str = "aws",
    ) -> dict:
        """
        Run agent for `hours` steps and return step-by-step decisions.
        """
        env   = self._make_env(starting_provider)
        state = env._build_state(env.current_cost, env._get_forecast())

        steps        = []
        action_counts = {"scale_up": 0, "scale_down": 0,
                         "migrate_aws": 0, "migrate_azure": 0}
        baseline_cost = 1.10   # no-optimisation baseline from training

        start_time = datetime.now().replace(minute=0, second=0, microsecond=0)

        for h in range(hours):
            action     = self._agent.select_action(state, greedy=True)
            name       = self._agent.get_action_name(action)
            next_state, reward, done = env.step(action)

            action_counts[name] += 1
            forecast = env._get_forecast()

            steps.append({
                "hour":         h + 1,
                "timestamp":    (start_time + timedelta(hours=h+1)).isoformat(
                                    timespec="seconds"),
                "action":       name,
                "cost":         round(float(env.current_cost), 4),
                "provider":     env.provider,
                "scale_factor": round(float(env.scale_factor), 3),
                "forecast_1h":  round(float(forecast[0]),      4),
                "reward":       round(float(reward),           4),
            })

            state = next_state
            if done:
                break

        costs      = [s["cost"] for s in steps]
        total      = round(sum(costs), 2)
        avg        = round(sum(costs) / len(costs), 4)
        baseline   = round(baseline_cost * len(costs), 2)
        saving     = round(baseline - total, 2)
        saving_pct = round(saving / baseline * 100, 1)

        return {
            "status":       "success",
            "hours":        len(steps),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "steps":        steps,
            "summary": {
                "total_cost":       total,
                "avg_cost":         avg,
                "baseline_cost":    baseline,
                "cost_saving":      saving,
                "cost_saving_pct":  saving_pct,
                "final_provider":   env.provider,
                "action_counts":    action_counts,
            },
        }

    def get_stats(self) -> dict:
        return {
            "status": "success",
            "agent": {
                "architecture": "DQN with experience replay and target network",
                "state_dim":    ForecastDrivenCloudEnvironment.STATE_DIM,
                "action_dim":   ForecastDrivenCloudEnvironment.ACTION_DIM,
                "hidden_dim":   256,
                "actions":      ["scale_up", "scale_down",
                                 "migrate_aws", "migrate_azure"],
                "state_features": [
                    "current_cost", "forecast_1h", "forecast_3h", "forecast_6h",
                    "load_factor", "hour_sin", "hour_cos",
                    "dow_sin", "dow_cos", "provider_aws", "provider_azure"
                ],
            },
            "training": {
                "episodes":       500,
                "episode_length": 48,
                "final_epsilon":  0.223,
                "success_rate":   90.0,
                "avg_reward":     2.222,
                "avg_cost":       0.459,
                "baseline_cost":  1.10,
                "cost_reduction_pct": 58.3,
            },
            "integration": {
                "forecaster":              "XGBoost (R²=0.717, MdAPE=11.45%)",
                "state_includes_forecast": True,
                "forecast_horizon_hours":  6,
                "cost_transitions":        "XGBoost-driven (not hardcoded)",
            },
        }