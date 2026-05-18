# Cloud Opti-X Backend

## Description
This repository contains the backend code for the Cloud Opti-X project, a multi-cloud cost optimization API. The backend uses machine learning models (XGBoost for forecasting and DQN for reinforcement learning) to optimize cloud resource costs and provide actionable recommendations.

## Tech Stack
- **Framework**: Flask 3.1.2
- **Language**: Python
- **ML/Data**: TensorFlow/PyTorch 2.9.1, scikit-learn 1.8.0, pandas 2.3.3, numpy 2.3.5
- **Visualization**: matplotlib 3.10.8, seaborn 0.13.2
- **Utilities**: python-dotenv, flask-cors 6.0.2

## Project Structure
```
cloud-opti-x-backend/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── routes/                   # API route handlers
│   ├── forecast.py          # Forecasting endpoints
│   ├── rl.py                # Reinforcement learning endpoints
│   └── alerts.py            # Budget alerts endpoints
├── services/                # Business logic services
├── utils/                   # Utility functions
├── data/                    # Data storage
├── trained_models/          # Serialized ML models
├── scripts/                 # Utility scripts
├── tests/                   # Test files
└── .gitignore              # Git ignore rules
```

## Features
- **Cost Forecasting**: XGBoost v2 Direct Multi-Horizon model for predicting cloud costs (R² = 0.907)
- **RL Recommendations**: DQN-based reinforcement learning agent for optimization recommendations
- **Multi-Cloud Support**: AWS and Azure comparison capabilities
- **Budget Alerts**: Real-time budget tracking and alert system
- **Model Explainability**: SHAP-based feature importance and explanation endpoints
- **Health Checks**: Service status and model validation endpoints

## Installation

### Prerequisites
- Python 3.8+
- pip

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/RanugiPerera/cloud-opti-x-backend.git
   cd cloud-opti-x-backend
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Start the server:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### General
- `GET /` - API information and available endpoints

### Forecasting
- `POST /api/forecast` - Single provider cost forecast
- `GET /api/forecast/compare` - AWS vs Azure cost comparison
- `GET /api/forecast/stats` - Forecast model metadata
- `GET /api/forecast/test` - Forecast health check
- `GET /api/forecast/explain` - SHAP model explainability
- `GET /api/forecast/importance` - Global feature importance

### Reinforcement Learning
- `POST /api/rl/recommend` - RL action recommendations
- `GET /api/rl/simulate` - Full episode simulation
- `GET /api/rl/stats` - RL agent metadata
- `GET /api/rl/test` - RL health check

### Alerts
- `GET /api/alerts` - Retrieve budget alerts
- `POST /api/alerts/dismiss` - Dismiss an alert

## Environment Setup
Create a `.env` file in the root directory for environment-specific configuration (e.g., database URLs, API keys).

## Testing
Test files are located in the `tests/` directory. Run tests as needed using your preferred Python testing framework.

## Trained Models
Pre-trained ML models are stored in the `trained_models/` directory and are automatically loaded when the server starts.

## CORS Configuration
The API is configured to accept cross-origin requests from all origins for `/api/*` endpoints with appropriate headers for web applications.

## Development Notes
- The application automatically loads forecasting and RL models on startup
- If either model fails to load, an error message will be displayed but the server will continue running
- Live streaming uses HTTP polling with a 30-second interval for real-time data updates



---

*Last updated: 2026-05-18*
