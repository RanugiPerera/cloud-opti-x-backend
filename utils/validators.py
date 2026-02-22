from functools import wraps
from flask import request, jsonify
from utils.config import Config


def validate_json(f):
    """
    Decorator to validate JSON input
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        if not data:
            return jsonify({'error': 'Empty JSON body'}), 400

        return f(*args, **kwargs)

    return decorated_function


def validate_forecast_request(f):
    """
    Decorator to validate forecast request parameters
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = request.get_json()

        # check required fields exist
        cloud_provider = data.get('cloud_provider', 'aws').lower()
        service_type = data.get('service_type', 'vm').lower()
        forecast_hours = data.get('forecast_hours', Config.DEFAULT_FORECAST_HOURS)

        # validate cloud provider
        if cloud_provider not in Config.VALID_CLOUD_PROVIDERS:
            return jsonify({
                'error': f'Invalid cloud_provider. Must be one of: {Config.VALID_CLOUD_PROVIDERS}'
            }), 400

        # validate service type
        if service_type not in Config.VALID_SERVICE_TYPES:
            return jsonify({
                'error': f'Invalid service_type. Must be one of: {Config.VALID_SERVICE_TYPES}'
            }), 400

        # validate forecast days
        if not isinstance(forecast_hours,
                          int) or forecast_hours < Config.MIN_FORECAST_HOURS or forecast_hours > Config.MAX_FORECAST_HOURS:
            return jsonify({
                'error': f'forecast_hours must be an integer between {Config.MIN_FORECAST_HOURS} and {Config.MAX_FORECAST_HOURS}'
            }), 400

        return f(*args, **kwargs)

    return decorated_function


def validate_required_fields(required_fields):
    """
    Decorator to validate required fields exist in JSON

    Usage:
        @validate_required_fields(['cloud_provider', 'service_type'])
        def my_endpoint():
            ...
    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            data = request.get_json()

            missing = [field for field in required_fields if field not in data]

            if missing:
                return jsonify({
                    'error': f'Missing required fields: {", ".join(missing)}'
                }), 400

            return f(*args, **kwargs)

        return decorated_function

    return decorator