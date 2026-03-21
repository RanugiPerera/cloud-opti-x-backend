import logging
from datetime import datetime
from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

bp = Blueprint("alerts", __name__, url_prefix="/api")

# In-memory dismissed alerts (resets on server restart — fine for demo)
_dismissed: set = set()

# Lazy service imports
_forecast_service = None
_rl_service       = None


def _get_forecast():
    global _forecast_service
    if _forecast_service is None:
        from services.forecast_service import ForecastService
        _forecast_service = ForecastService()
    return _forecast_service


def _get_rl():
    global _rl_service
    if _rl_service is None:
        from services.rl_service import RLService
        _rl_service = RLService()
    return _rl_service


@bp.route("/alerts", methods=["GET"])
def get_alerts():
    
    forecast_hours = request.args.get("forecast_hours", 24, type=int)
    forecast_hours = max(1, min(48, forecast_hours))

    try:
        svc       = _get_forecast()
        threshold = request.args.get(
            "threshold", svc._budget, type=float
        )

        alerts = []

        # Fetch AWS and Azure forecasts in parallel
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=2) as ex:
            f_aws   = ex.submit(svc.predict_costs, "aws",   forecast_hours)
            f_azure = ex.submit(svc.predict_costs, "azure", forecast_hours)
            provider_results = {
                "aws":   f_aws.result(),
                "azure": f_azure.result(),
            }

        for provider in ("aws", "azure"):
            result = provider_results[provider]
            costs  = result["costs"]
            times  = result["timestamps"]

            for i, (cost, ts) in enumerate(zip(costs, times)):
                if cost <= threshold:
                    continue

                alert_id = f"{provider}-{i+1}"
                if alert_id in _dismissed:
                    continue

                overage  = cost - threshold
                severity = (
                    "critical" if overage > threshold * 0.3 else
                    "high"     if overage > threshold * 0.1 else
                    "medium"
                )

                # Ask the RL agent what to do
                rl_action   = "scale_down"
                rl_reasoning = "Reduce resource allocation to bring costs below budget."
                try:
                    rl_result    = _get_rl().recommend(
                        current_cost=cost,
                        provider=provider,
                        scale_factor=1.0,
                    )
                    rl_action    = rl_result["action"]
                    rl_reasoning = rl_result["reasoning"]
                except Exception:
                    pass   # RL agent optional — alerts still useful without it

                alerts.append({
                    "id":             alert_id,
                    "hour":           i + 1,
                    "timestamp":      ts,
                    "provider":       provider,
                    "predicted_cost": round(cost, 4),
                    "threshold":      round(threshold, 4),
                    "overage":        round(overage, 4),
                    "severity":       severity,
                    "rl_action":      rl_action,
                    "rl_reasoning":   rl_reasoning,
                    "dismissed":      False,
                })

        # Sort by severity then overage
        severity_order = {"critical": 0, "high": 1, "medium": 2}
        alerts.sort(key=lambda a: (severity_order[a["severity"]], -a["overage"]))

        return jsonify({
            "status":       "success",
            "threshold":    round(threshold, 4),
            "forecast_hours": forecast_hours,
            "total_alerts": len(alerts),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "alerts":       alerts,
        }), 200

    except Exception as exc:
        logger.exception("Alerts failed")
        return jsonify({"error": str(exc)}), 500


@bp.route("/alerts/dismiss", methods=["POST"])
def dismiss_alert():

    data     = request.get_json(silent=True) or {}
    alert_id = data.get("alert_id")

    if not alert_id:
        return jsonify({"error": "alert_id is required"}), 400

    _dismissed.add(alert_id)
    return jsonify({
        "status":   "success",
        "dismissed": alert_id,
        "message":  f"Alert {alert_id} dismissed until server restart",
    }), 200


@bp.route("/alerts/clear", methods=["POST"])
def clear_dismissed():
    _dismissed.clear()
    return jsonify({"status": "success", "message": "All dismissed alerts restored"}), 200