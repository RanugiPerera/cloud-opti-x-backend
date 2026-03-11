from flask import Flask, jsonify
from flask_cors import CORS

# import routes
from routes import forecast, rl

app = Flask(__name__)

# enable CORS for frontend
CORS(app)

# register routes
app.register_blueprint(forecast.bp)
app.register_blueprint(rl.bp)


# welcome endpoint
@app.route('/')
def index():
    return jsonify({
        "service":     "Multi-Cloud Cost Optimization API",
        "version":     "2.0.0",
        "status":      "running",
        "model":       "XGBoost (R²=0.717, MdAPE=11.45%)",
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


if __name__ == '__main__':
    print("=" * 60)
    print("  Multi-Cloud Cost Optimization API")
    print("  XGBoost Forecaster + DQN RL Agent")
    print("=" * 60)
    print("Starting server on http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET  /                          - API info")
    print("  POST /api/forecast              - Single provider forecast")
    print("  GET  /api/forecast/compare      - AWS vs Azure comparison")
    print("  GET  /api/forecast/stats        - Forecast model metadata")
    print("  GET  /api/forecast/test         - Forecast health check")
    print("  POST /api/rl/recommend          - RL action recommendation")
    print("  GET  /api/rl/simulate           - Full episode simulation")
    print("  GET  /api/rl/stats              - RL agent metadata")
    print("  GET  /api/rl/test               - RL health check")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=True)