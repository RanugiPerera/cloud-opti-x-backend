"""
RL Agent API Blueprint
======================
Flask routes that expose the DQN reinforcement learning agent via REST API.

Endpoints
---------
POST /api/rl/recommend          — get action recommendation for current state
GET  /api/rl/simulate           — run a full episode and return step-by-step decisions
GET  /api/rl/stats              — agent metadata and training summary
GET  /api/rl/test               — health check

"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

bp = Blueprint("rl", __name__, url_prefix="/api/rl")

# Lazy singleton — initialised on first request
_rl_service = None


def get_rl_service():
    global _rl_service
    if _rl_service is None:
        try:
            from services.rl_service import RLService
            _rl_service = RLService()
            logger.info("RLService initialised successfully")
        except Exception as e:
            logger.error(f"RLService failed to initialise: {e}")
            raise
    return _rl_service


# ============================================================================
# POST /api/rl/recommend
# ============================================================================

@bp.route("/recommend", methods=["POST"])
def recommend():
    """
    Get the agent's recommended action given a current cloud state.

    The agent observes the current cost + XGBoost 6-hour forecast and
    returns the optimal action according to the learned policy.

    Request body (JSON)
    -------------------
    {
        "current_cost":   1.20,        (current $/hr — required)
        "provider":       "aws",       (current provider — default: "aws")
        "scale_factor":   1.0          (current resource scale — default: 1.0)
    }

    Response
    --------
    {
        "status":          "success",
        "action":          "scale_down",
        "action_id":       1,
        "reasoning":       "Forecast shows costs declining — scale down to reduce spend",
        "current_state": {
            "current_cost":   1.20,
            "provider":       "aws",
            "scale_factor":   1.0,
            "forecast_1h":    1.15,
            "forecast_3h":    0.95,
            "forecast_6h":    0.72
        },
        "expected_outcome": {
            "estimated_cost_after": 0.90,
            "cost_saving_pct":      25.0
        }
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    current_cost = data.get("current_cost")
    if current_cost is None:
        return jsonify({"error": "current_cost is required"}), 400

    try:
        current_cost = float(current_cost)
    except (TypeError, ValueError):
        return jsonify({"error": "current_cost must be a number"}), 400

    if current_cost <= 0:
        return jsonify({"error": "current_cost must be positive"}), 400

    provider     = str(data.get("provider",     "aws")).lower()
    scale_factor = float(data.get("scale_factor", 1.0))

    if provider not in ("aws", "azure"):
        return jsonify({"error": "provider must be 'aws' or 'azure'"}), 400

    try:
        result = get_rl_service().recommend(
            current_cost = current_cost,
            provider     = provider,
            scale_factor = scale_factor,
        )
    except Exception as exc:
        logger.exception("Recommend failed")
        return jsonify({"error": str(exc)}), 500

    return jsonify(result), 200


# ============================================================================
# GET /api/rl/simulate
# ============================================================================

@bp.route("/simulate", methods=["GET"])
def simulate():
    """
    Run the agent through a full episode and return step-by-step decisions.

    This is the key demo endpoint — it shows the agent making 48 sequential
    decisions using the XGBoost forecast at each step.

    """
    hours    = request.args.get("hours",    24,    type=int)
    provider = request.args.get("provider", "aws", type=str).lower()

    if not (1 <= hours <= 48):
        return jsonify({"error": "hours must be between 1 and 48"}), 400

    if provider not in ("aws", "azure"):
        return jsonify({"error": "provider must be 'aws' or 'azure'"}), 400

    try:
        result = get_rl_service().simulate_episode(
            hours            = hours,
            starting_provider = provider,
        )
    except Exception as exc:
        logger.exception("Simulation failed")
        return jsonify({"error": str(exc)}), 500

    return jsonify(result), 200


# ============================================================================
# GET /api/rl/stats
# ============================================================================

@bp.route("/stats", methods=["GET"])
def get_stats():
    """
    Return RL agent metadata and training summary.
    """
    try:
        stats = get_rl_service().get_stats()
    except Exception as exc:
        logger.exception("Stats failed")
        return jsonify({"error": str(exc)}), 500

    return jsonify(stats), 200


# ============================================================================
# GET /api/rl/test
# ============================================================================

@bp.route("/test", methods=["GET"])
def test_agent():
    """
    Health check — verifies the RL agent loads and produces valid actions.

    """
    try:
        result = get_rl_service().recommend(
            current_cost = 1.10,
            provider     = "aws",
            scale_factor = 1.0,
        )
        return jsonify({
            "status":        "success",
            "message":       "DQN agent is operational",
            "sample_action": result["action"],
        }), 200

    except Exception as exc:
        logger.exception("RL health check failed")
        return jsonify({
            "status":  "error",
            "message": "DQN agent failed to run",
            "detail":  str(exc),
        }), 500
