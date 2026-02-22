from flask import Blueprint, request, jsonify
from services.forecast_service import forecast_service

bp = Blueprint('forecast', __name__, url_prefix='/api')


@bp.route('/forecast', methods=['POST'])
def forecast_costs():
    """
    Forecast future cloud costs using Transformer model

    Expected input:
    {
        "cloud_provider": "aws" or "azure",
        "service_type": "vm", "storage", "network", or "kubernetes",
        "forecast_hours": 24 (hours to forecast)
    }

    Returns:
    {
        "status": "success",
        "predictions": [list of predicted costs],
        "time_labels": ["+1h", "+2h", ...],
        "confidence_interval": {"lower": [...], "upper": [...]}
    }
    """

    # get request data
    data = request.get_json()

    # basic validation
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # extract parameters with defaults
    cloud_provider = data.get('cloud_provider', 'aws').lower()
    service_type = data.get('service_type', 'vm').lower()
    forecast_hours = data.get('forecast_hours', 24)

    # validate parameters
    valid_providers = ['aws', 'azure']
    valid_services = ['vm', 'storage', 'network', 'kubernetes']

    if cloud_provider not in valid_providers:
        return jsonify({
            "error": f"Invalid cloud_provider. Must be one of: {valid_providers}"
        }), 400

    if service_type not in valid_services:
        return jsonify({
            "error": f"Invalid service_type. Must be one of: {valid_services}"
        }), 400

    if not isinstance(forecast_hours, int) or forecast_hours < 1 or forecast_hours > 24:
        return jsonify({
            "error": "forecast_hours must be an integer between 1 and 24"
        }), 400

    # make prediction
    result = forecast_service.predict_costs(
        cloud_provider=cloud_provider,
        service_type=service_type,
        forecast_hours=forecast_hours
    )

    # check for errors
    if 'error' in result:
        return jsonify(result), 500

    return jsonify(result), 200


@bp.route('/forecast/stats', methods=['GET'])
def get_forecast_stats():
    """
    Get statistics about the forecasting model

    Returns model info and historical data summary
    """

    stats = forecast_service.get_summary_stats()

    if 'error' in stats:
        return jsonify(stats), 500

    return jsonify(stats), 200


@bp.route('/forecast/test', methods=['GET'])
def test_forecast():
    """
    Quick test endpoint to verify model is working
    """

    # run a test prediction
    result = forecast_service.predict_costs(
        cloud_provider='aws',
        service_type='vm',
        forecast_hours=24
    )

    if 'error' in result:
        return jsonify({
            'status': 'error',
            'message': 'Model not working',
            'details': result
        }), 500

    return jsonify({
        'status': 'success',
        'message': 'Forecast model is working!',
        'sample_prediction': result
    }), 200