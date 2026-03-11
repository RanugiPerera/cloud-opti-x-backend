"""
Forecast API Blueprint
Flask routes that expose the XGBoost cost forecasting model via REST API.

Endpoints
---------
POST /api/forecast          — generate N-hour cost forecast
GET  /api/forecast/compare  — side-by-side AWS vs Azure comparison
GET  /api/forecast/stats    — model info and cost history summary
GET  /api/forecast/test     — health check
"""

import logging
from datetime import datetime
from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

_service = None

def get_service():
    """Return the ForecastService singleton, initialising it on first call."""
    global _service
    if _service is None:
        try:
            from services.forecast_service import ForecastService
            _service = ForecastService()
            logger.info("ForecastService initialised successfully")
        except Exception as e:
            logger.error(f"ForecastService failed to initialise: {e}")
            raise
    return _service

bp = Blueprint("forecast", __name__, url_prefix="/api")


# ============================================================================
# POST /api/forecast
# ============================================================================

@bp.route("/forecast", methods=["POST"])
def forecast_costs():

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    # ── Validate inputs ───────────────────────────────────────────────────────
    cloud_provider = str(data.get("cloud_provider", "aws")).lower()
    service_type   = str(data.get("service_type",   "vm")).lower()
    forecast_hours = data.get("forecast_hours", 24)

    if cloud_provider not in ("aws", "azure"):
        return jsonify({
            "error": "cloud_provider must be 'aws' or 'azure'"
        }), 400

    if service_type not in ("vm", "storage", "network", "kubernetes"):
        return jsonify({
            "error": "service_type must be one of: vm, storage, network, kubernetes"
        }), 400

    if not isinstance(forecast_hours, int) or not (1 <= forecast_hours <= 48):
        return jsonify({
            "error": "forecast_hours must be an integer between 1 and 48"
        }), 400

    # ── Run forecast ──────────────────────────────────────────────────────────
    try:
        result = get_service().predict_costs(
            cloud_provider = cloud_provider,
            forecast_hours = forecast_hours,
        )
    except Exception as exc:
        logger.exception("Forecast failed")
        return jsonify({"error": str(exc)}), 500

    # ── Format response ───────────────────────────────────────────────────────
    predictions = []
    for i, (ts, cost) in enumerate(
        zip(result["timestamps"], result["costs"][:forecast_hours])
    ):
        predictions.append({
            "hour":      i + 1,
            "timestamp": ts,
            "cost":      round(cost, 4),
            "lower":     round(cost * 0.85, 4),   # ±15% band from MdAPE
            "upper":     round(cost * 1.15, 4),
        })

    costs          = [p["cost"] for p in predictions]
    budget_thresh  = result.get("budget_threshold", 1.49)

    return jsonify({
        "status":         "success",
        "provider":       cloud_provider,
        "service_type":   service_type,
        "forecast_hours": forecast_hours,
        "model":          "XGBoost (R²=0.717, MdAPE=11.45%)",
        "generated_at":   datetime.now().isoformat(timespec="seconds"),
        "predictions":    predictions,
        "summary": {
            "total_cost":         round(sum(costs), 2),
            "avg_cost":           round(sum(costs) / len(costs), 4),
            "min_cost":           round(min(costs), 4),
            "max_cost":           round(max(costs), 4),
            "budget_threshold":   round(budget_thresh, 4),
            "hours_over_budget":  sum(1 for c in costs if c > budget_thresh),
        },
    }), 200


# ============================================================================
# GET /api/forecast/compare
# ============================================================================

@bp.route("/forecast/compare", methods=["GET"])
def compare_providers():
    """
    Generate side-by-side AWS vs Azure forecast and return cost comparison.
"""
    forecast_hours = request.args.get("forecast_hours", 48, type=int)
    if not (1 <= forecast_hours <= 48):
        return jsonify({
            "error": "forecast_hours must be between 1 and 48"
        }), 400

    try:
        aws_result   = get_service().predict_costs("aws",   forecast_hours)
        azure_result = get_service().predict_costs("azure", forecast_hours)
    except Exception as exc:
        logger.exception("Compare forecast failed")
        return jsonify({"error": str(exc)}), 500

    def format_predictions(result, hours):
        out = []
        for i, (ts, cost) in enumerate(
            zip(result["timestamps"], result["costs"][:hours])
        ):
            out.append({
                "hour":      i + 1,
                "timestamp": ts,
                "cost":      round(cost, 4),
                "lower":     round(cost * 0.85, 4),
                "upper":     round(cost * 1.15, 4),
            })
        return out

    aws_preds   = format_predictions(aws_result,   forecast_hours)
    azure_preds = format_predictions(azure_result, forecast_hours)

    aws_total   = round(sum(p["cost"] for p in aws_preds),   2)
    azure_total = round(sum(p["cost"] for p in azure_preds), 2)
    cheaper     = "aws" if aws_total < azure_total else "azure"
    saving      = round(abs(aws_total - azure_total), 2)
    saving_pct  = round(saving / max(aws_total, azure_total) * 100, 1)

    return jsonify({
        "status":         "success",
        "forecast_hours": forecast_hours,
        "model":          "XGBoost (R²=0.717, MdAPE=11.45%)",
        "generated_at":   datetime.now().isoformat(timespec="seconds"),
        "aws": {
            "predictions": aws_preds,
            "total_cost":  aws_total,
            "avg_cost":    round(aws_total / forecast_hours, 4),
        },
        "azure": {
            "predictions": azure_preds,
            "total_cost":  azure_total,
            "avg_cost":    round(azure_total / forecast_hours, 4),
        },
        "comparison": {
            "cheaper_provider": cheaper,
            "saving":           saving,
            "saving_pct":       saving_pct,
            "recommendation":   (
                f"Use {cheaper.upper()} — "
                f"saves ${saving} over {forecast_hours} hours ({saving_pct}%)"
            ),
        },
    }), 200


# ============================================================================
# GET /api/forecast/stats
# ============================================================================

@bp.route("/forecast/stats", methods=["GET"])
def get_forecast_stats():

    try:
        stats = get_service().get_stats()
    except Exception as exc:
        logger.exception("Stats failed")
        return jsonify({"error": str(exc)}), 500

    return jsonify(stats), 200


# ============================================================================
# GET /api/forecast/test
# ============================================================================

@bp.route("/forecast/test", methods=["GET"])
def test_forecast():
    """
    Health check — verifies the model loads and produces valid predictions.
    """
    try:
        result = get_service().predict_costs("aws", forecast_hours=24)
        costs  = result["costs"]

        return jsonify({
            "status":  "success",
            "message": "XGBoost forecast model is operational",
            "model":   "XGBoost (R²=0.717, MdAPE=11.45%)",
            "sample": {
                "provider":    "aws",
                "hour_1_cost":  round(costs[0],  4),
                "hour_24_cost": round(costs[23], 4),
            },
        }), 200

    except Exception as exc:
        logger.exception("Health check failed")
        return jsonify({
            "status":  "error",
            "message": "Forecast model failed to run",
            "detail":  str(exc),
        }), 500
