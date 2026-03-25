from flask import Flask, jsonify
from flask_cors import CORS

# import routes
from routes import forecast, rl
from routes.alerts import bp as alerts_bp

app = Flask(__name__)

CORS(app, resources={r"/api/*": {
    "origins":  "*",
    "methods":  ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization", "Accept"],
}})

# register routes
app.register_blueprint(forecast.bp)
app.register_blueprint(rl.bp)
app.register_blueprint(alerts_bp)

print("Loading models — please wait...")

with app.app_context():
    try:
        from routes.forecast import get_service
        get_service()
        print("✓ ForecastService ready")
    except Exception as e:
        print(f"✗ ForecastService failed: {e}")

    try:
        from routes.rl import get_rl_service
        get_rl_service()
        print("✓ RLService ready")
    except Exception as e:
        print(f"✗ RLService failed: {e}")

print("Models loaded — starting server...")


# welcome endpoint
@app.route('/')
def index():
    return jsonify({
        "service":     "Multi-Cloud Cost Optimization API",
        "version":     "2.0.0",
        "status":      "running",
        "model":       "XGBoost v2 Direct Multi-Horizon (R²_log=0.907)",
        "endpoints": {
            "forecast":         "POST /api/forecast",
            "compare":          "GET  /api/forecast/compare",
            "forecast_stats":   "GET  /api/forecast/stats",
            "forecast_test":    "GET  /api/forecast/test",
            "rl_recommend":     "POST /api/rl/recommend",
            "rl_simulate":      "GET  /api/rl/simulate",
            "rl_stats":         "GET  /api/rl/stats",
            "rl_test":          "GET  /api/rl/test",
        }
    })


# global error handler
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# Live streaming uses HTTP polling from the frontend (LiveForecastChart.tsx)


if __name__ == '__main__':
    print("=" * 60)
    print("  Multi-Cloud Cost Optimization API")
    print("  XGBoost v2 Direct Multi-Horizon + DQN RL Agent")
    print("=" * 60)
    print("Starting server on http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET  /                          - API info")
    print("  POST /api/forecast              - Single provider forecast")
    print("  GET  /api/forecast/compare      - AWS vs Azure comparison")
    print("  GET  /api/forecast/stats        - Forecast model metadata")
    print("  GET  /api/forecast/test         - Forecast health check")
    print("  GET  /api/forecast/explain      - SHAP explainability [NEW]")
    print("  GET  /api/forecast/importance   - Global feature importance [NEW]")
    print("  POST /api/rl/recommend          - RL action recommendation")
    print("  GET  /api/rl/simulate           - Full episode simulation")
    print("  GET  /api/rl/stats              - RL agent metadata")
    print("  GET  /api/rl/test               - RL health check")
    print("  GET  /api/alerts                - Budget alerts [NEW]")
    print("  POST /api/alerts/dismiss        - Dismiss alert [NEW]")
    print("  WS   /                          - Live streaming (HTTP polling, 30s interval)")
    print("=" * 60)

    print("  Use 'python run.py' for WebSocket (eventlet) support")
    print("=" * 60)
    # REST API only when running app.py directly
    app.run(host='0.0.0.0', port=5000, debug=False)