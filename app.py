from flask import Flask, jsonify
from flask_cors import CORS

# import routes
from routes import forecast

app = Flask(__name__)

# enable CORS for frontend
CORS(app)

# register routes
app.register_blueprint(forecast.bp)



# welcome endpoint
@app.route('/')
def index():
    return jsonify({
        "service": "Multi-Cloud Cost Optimization API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "forecast": "/api/forecast",
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
    print(" Multi-Cloud Cost Optimization API")
    print("=" * 60)
    print("Starting server on http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET  /                      - API info")
    print("  POST /api/forecast          - Cost forecasting")
    print("=" * 60)

    app.run(host='0.0.0.0', port=5000, debug=True)