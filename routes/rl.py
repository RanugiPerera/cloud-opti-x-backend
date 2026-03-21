import logging
import sys
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

bp = Blueprint("rl", __name__, url_prefix="/api/rl")

import threading as _threading

_rl_service       = None
_rl_service_lock  = _threading.Lock()
_rl_service_event = _threading.Event()

def get_rl_service():
    """Return the RLService singleton using Event-based pattern."""
    global _rl_service

    if _rl_service is not None:
        return _rl_service

    with _rl_service_lock:
        if _rl_service is None and not _rl_service_event.is_set():
            try:
                from services.rl_service import RLService
                instance    = RLService()
                _rl_service = instance
                logger.info("RLService initialised successfully")
            except Exception as e:
                logger.error(f"RLService failed to initialise: {e}")
                _rl_service_event.set()
                raise
            finally:
                _rl_service_event.set()

    _rl_service_event.wait(timeout=300)

    if _rl_service is None:
        raise RuntimeError("RLService failed to initialise after 300s")
    return _rl_service


# ============================================================================
# POST /api/rl/recommend
# ============================================================================

@bp.route("/recommend", methods=["POST"])
def recommend():
   
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

    try:
        svc = get_rl_service()

        # Lightweight check — verify agent network is loaded.
        # Do NOT call recommend() here — that runs a full XGBoost
        # forward pass on every health check.
        import torch
        n_params = sum(p.numel() for p in svc._agent.q_network.parameters())

        return jsonify({
            "status":   "success",
            "message":  "DQN agent is operational",
            "n_params": n_params,
            "actions":  ["scale_up", "scale_down", "migrate_aws", "migrate_azure"],
        }), 200

    except Exception as exc:
        logger.exception("RL health check failed")
        return jsonify({
            "status":  "error",
            "message": "DQN agent failed to load",
            "detail":  str(exc),
        }), 500
