import numpy as np
import torch
from scripts.models.rl_agent_integrated import DQNAgent, ForecastDrivenCloudEnvironment

def test_dqn_forward_pass():
    agent = DQNAgent(state_dim=11, action_dim=4)
    state = torch.FloatTensor(np.random.randn(11))
    with torch.no_grad():
        q_values = agent.q_network(state)
    assert q_values.shape == (4,)
    assert not torch.isnan(q_values).any()
    
import xgboost as xgb

class MockXGB:
    def get_booster(self):
        class Booster:
            @property
            def feature_names(self):
                return [f"f{i}" for i in range(61)]
        return Booster()
    def predict(self, X):
        return np.zeros(X.shape[0] if hasattr(X, "shape") else 1)

def test_state_vector_dimensions():
    model = MockXGB()
    cost_history = np.random.randn(200).tolist()
    env   = ForecastDrivenCloudEnvironment(xgb_model=model, cost_history=cost_history)
    state = env.reset()
    # State: [current_cost, forecast_1h, forecast_3h, forecast_6h,
    #         load_factor, hour_sin, hour_cos, dow_sin, dow_cos,
    #         provider_aws, provider_azure]
    assert len(state) == 11
    assert all(np.isfinite(state))
    
def test_rl_recommendation_format():
    from routes.alerts import _get_rl
    rl = _get_rl()
    recommendation = rl.recommend(current_cost=1.0, provider="aws", scale_factor=1.0)
    assert "action" in recommendation
    assert "reasoning" in recommendation
    assert recommendation["action"] in ("scale_down", "hold", "scale_up")
    assert isinstance(recommendation["reasoning"], str)
    
    
def test_all_actions_valid():
    model = MockXGB()
    cost_history = np.random.randn(200).tolist()
    env = ForecastDrivenCloudEnvironment(xgb_model=model, cost_history=cost_history)
    for action in range(4):
        state = env.reset()
        next_state, reward, done = env.step(action)
        assert len(next_state) == 11
        assert np.isfinite(reward)
        assert isinstance(done, bool)