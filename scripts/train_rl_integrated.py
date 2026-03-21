"""
Integrated RL Training Script
==============================
Trains the DQN agent inside the ForecastDrivenCloudEnvironment,
where every cost transition is driven by the XGBoost forecaster.

Run order
---------
1. python scripts/full_pipeline_v2.py       — train v2 XGBoost forecaster
2. python scripts/train_rl_integrated.py    — train RL agent (this file)
3. python scripts/forecast_demo.py          — generate demo charts

What this demonstrates for your thesis
---------------------------------------
The RL agent learns a policy over the XGBoost forecast horizon:
  - State includes the 1h/3h/6h cost forecast → agent plans ahead
  - Actions affect scale_factor and provider → directly drives cost
  - Reward penalises cost, rewards anticipatory provider switching
  - The two models are tightly coupled: XGBoost forecasts, RL decides
"""

import glob
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "trained_models"
PLOTS_DIR  = BASE_DIR / "data" / "processed"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Robust import: find rl_agent_integrated.py wherever it lives ─────────────
# Searches these locations in order:
#   1. project_root/models/
#   2. same folder as this script  (scripts/)
#   3. scripts/models/
#   4. project root itself
_SCRIPT_DIR = Path(__file__).resolve().parent
_candidates = [
    BASE_DIR / "models",          # correct location
    _SCRIPT_DIR,                  # scripts/
    _SCRIPT_DIR / "models",       # scripts/models/  (where it currently is)
    BASE_DIR,                     # project root
]
for _p in _candidates:
    if (_p / "rl_agent_integrated.py").exists():
        sys.path.insert(0, str(_p))
        break
else:
    print("ERROR: rl_agent_integrated.py not found in any expected location.")
    print("Searched:", [str(p) for p in _candidates])
    sys.exit(1)

from rl_agent_integrated import DQNAgent, ForecastDrivenCloudEnvironment

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("rl_training")

# ── Hyperparameters ───────────────────────────────────────────────────────────
NUM_EPISODES       = 500
EPISODE_LENGTH     = 48      # 48 hours per episode
BATCH_SIZE         = 64
UPDATE_TARGET_EVERY = 10
WARMUP_EPISODES    = 20      # episodes before we start logging meaningful stats
DEVICE             = "cpu"


# ============================================================================
# STEP 1: Load XGBoost model
# ============================================================================

def load_xgb_model() -> xgb.XGBRegressor:
    """
    Load the +1h q50 model from v2 pipeline (direct multi-horizon).
    Falls back to v1 single model if v2 models are not found.
    The +1h model is used for the RL environment because the agent
    makes 1-step transitions — it needs 1-hour-ahead cost predictions.
    """
    # v2: look for horizon-specific q50 model (point forecast)
    v2_files = sorted(glob.glob(str(MODELS_DIR / "xgb_h1h_q50_*.json")))
    if v2_files:
        path = v2_files[-1]
        logger.info(f"Loading v2 direct +1h model: {Path(path).name}")
    else:
        # fallback to v1 single model
        v1_files = sorted(glob.glob(str(MODELS_DIR / "xgb_cost_model_*.json")))
        if not v1_files:
            raise FileNotFoundError(
                f"No XGBoost model found in {MODELS_DIR}.\n"
                "Run full_pipeline_v2.py first to train v2 models."
            )
        path = v1_files[-1]
        logger.info(f"v2 models not found — loading v1 fallback: {Path(path).name}")

    model = xgb.XGBRegressor()
    model.load_model(path)
    feat_names = model.get_booster().feature_names
    n_feat = len(feat_names) if feat_names else model.get_booster().num_features()
    logger.info(f"XGBoost model loaded: {Path(path).name} ({n_feat} features)")
    return model


# ============================================================================
# STEP 2: Build cost history for environment initialisation
# ============================================================================

def build_cost_history(n: int = 2000) -> np.ndarray:
    """
    Reconstruct a realistic cost history using the same log-space model
    as the training pipeline. This gives the RL environment a proper
    cost buffer so lag features are well-initialised from episode start.
    """
    rng      = np.random.default_rng(42)
    phi, sigma = 0.70, 0.06

    times    = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(n)]
    hours    = np.array([t.hour      for t in times])
    dows     = np.array([t.weekday() for t in times])
    months   = np.array([t.month     for t in times])

    diurnal = (0.35 * np.exp(-0.5 * ((hours - 14) / 3.5) ** 2)
             + 0.15 * np.exp(-0.5 * ((hours -  9) / 2.0) ** 2)
             - 0.25 * np.exp(-0.5 * ((hours -  4) / 2.5) ** 2))
    weekly  = np.where(dows < 5, 0.05, -0.55)
    trend   = (months - 1) * np.log(1.008)

    ar = np.zeros(n)
    eps = rng.normal(0, sigma, n)
    for t in range(1, n):
        ar[t] = phi * ar[t-1] + eps[t]

    log_cost = diurnal + weekly + trend + ar + rng.normal(0, 0.10, n)
    return np.clip(np.exp(log_cost), 0.05, None)


# ============================================================================
# STEP 3: Training loop
# ============================================================================

def train(
    agent: DQNAgent,
    env:   ForecastDrivenCloudEnvironment,
) -> dict:
    episode_rewards  = []
    episode_costs    = []
    episode_losses   = []
    action_totals    = {i: 0 for i in range(4)}
    success_count    = 0

    logger.info("=" * 62)
    logger.info("  INTEGRATED RL TRAINING  —  XGBoost-Driven Environment")
    logger.info("=" * 62)
    logger.info(f"  Episodes     : {NUM_EPISODES}")
    logger.info(f"  Episode len  : {EPISODE_LENGTH} hours")
    logger.info(f"  State dim    : {ForecastDrivenCloudEnvironment.STATE_DIM}")
    logger.info(f"  Actions      : {agent.action_names}")
    logger.info("=" * 62)

    for episode in range(NUM_EPISODES):
        state          = env.reset()
        ep_reward      = 0.0
        ep_losses      = []
        ep_costs       = []

        for step in range(EPISODE_LENGTH):
            action = agent.select_action(state)
            action_totals[action] += 1

            next_state, reward, done = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)

            loss = agent.update(BATCH_SIZE)
            if loss is not None:
                ep_losses.append(loss)

            ep_reward += reward
            ep_costs.append(env.current_cost)
            state = next_state

            if done:
                break

        if episode % UPDATE_TARGET_EVERY == 0:
            agent.update_target_network()

        agent.decay_epsilon()

        avg_cost = float(np.mean(ep_costs)) if ep_costs else 0.0
        if avg_cost < 1.0:
            success_count += 1

        episode_rewards.append(ep_reward)
        episode_costs.append(avg_cost)
        episode_losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)

        if (episode + 1) % 50 == 0:
            recent = slice(max(0, episode - 49), episode + 1)
            logger.info(
                f"Ep {episode+1:>4}/{NUM_EPISODES}  "
                f"AvgReward={np.mean(episode_rewards[recent]):>7.3f}  "
                f"AvgCost=${np.mean(episode_costs[recent]):>5.3f}  "
                f"Success={success_count/(episode+1)*100:>4.1f}%  "
                f"ε={agent.epsilon:.3f}  "
                f"Loss={np.mean(episode_losses[recent]):.4f}"
            )

    return dict(
        rewards=episode_rewards,
        costs=episode_costs,
        losses=episode_losses,
        action_totals=action_totals,
        success_count=success_count,
    )


# ============================================================================
# STEP 4: Evaluation
# ============================================================================

def evaluate(
    agent: DQNAgent,
    env:   ForecastDrivenCloudEnvironment,
    n_episodes: int = 20,
) -> None:
    logger.info("\n" + "=" * 62)
    logger.info("  EVALUATION  —  Greedy Policy (ε = 0)")
    logger.info("=" * 62)

    test_rewards, test_costs = [], []
    action_counts = {i: 0 for i in range(4)}
    provider_counts = {"aws": 0, "azure": 0}

    for ep in range(n_episodes):
        state = env.reset()
        ep_reward, ep_costs_list = 0.0, []

        for step in range(EPISODE_LENGTH):
            action = agent.select_action(state, greedy=True)
            action_counts[action] += 1
            provider_counts[env.provider] += 1
            next_state, reward, done = env.step(action)
            ep_reward += reward
            ep_costs_list.append(env.current_cost)
            state = next_state
            if done:
                break

        test_rewards.append(ep_reward)
        test_costs.append(float(np.mean(ep_costs_list)))
        logger.info(
            f"  Test ep {ep+1:>2}: reward={ep_reward:>7.3f}  "
            f"avg_cost=${test_costs[-1]:.3f}  "
            f"provider={env.provider}"
        )

    logger.info("\n" + "=" * 62)
    logger.info("  EVALUATION METRICS")
    logger.info("=" * 62)
    logger.info(f"  Avg reward   : {np.mean(test_rewards):.3f} ± {np.std(test_rewards):.3f}")
    logger.info(f"  Avg cost/hr  : ${np.mean(test_costs):.3f} ± ${np.std(test_costs):.3f}")
    logger.info(f"  Cost vs baseline ($1.10/hr): "
                f"{(1.10 - np.mean(test_costs)) / 1.10 * 100:+.1f}%")

    total_actions = sum(action_counts.values())
    logger.info(f"\n  Action distribution:")
    for a, cnt in action_counts.items():
        logger.info(f"    {agent.get_action_name(a):<15}: "
                    f"{cnt:>5} ({cnt/total_actions*100:.1f}%)")

    total_provider = sum(provider_counts.values())
    logger.info(f"\n  Provider preference:")
    for p, cnt in provider_counts.items():
        logger.info(f"    {p:<6}: {cnt/total_provider*100:.1f}% of steps")


# ============================================================================
# STEP 5: Plot training curves
# ============================================================================

def plot_training(stats: dict, output_path: Path) -> None:
    rewards = stats["rewards"]
    costs   = stats["costs"]
    losses  = stats["losses"]
    window  = 20

    def moving_avg(x):
        return np.convolve(x, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Integrated RL Training — DQN + XGBoost Cost Forecaster\n"
        "State includes XGBoost 1h/3h/6h forecast; "
        "transitions driven by forecaster",
        fontsize=11, fontweight="bold",
    )

    # Rewards
    ax = axes[0]
    ax.plot(rewards, alpha=0.25, color="steelblue", label="Episode reward")
    ax.plot(range(window-1, len(rewards)), moving_avg(rewards),
            color="steelblue", linewidth=2, label=f"{window}-ep moving avg")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
    ax.set_title("Training Rewards"); ax.legend(); ax.grid(alpha=0.3)

    # Costs
    ax = axes[1]
    ax.plot(costs, alpha=0.25, color="tomato", label="Avg cost/hr")
    ax.plot(range(window-1, len(costs)), moving_avg(costs),
            color="tomato", linewidth=2, label=f"{window}-ep moving avg")
    ax.axhline(1.10, color="grey", linestyle="--", linewidth=1,
               label="Baseline $1.10/hr")
    ax.set_xlabel("Episode"); ax.set_ylabel("Cost ($/hr)")
    ax.set_title("Episode Average Cost"); ax.legend(); ax.grid(alpha=0.3)

    # Losses
    ax = axes[2]
    ax.plot(losses, alpha=0.5, color="seagreen")
    ax.set_xlabel("Episode"); ax.set_ylabel("MSE Loss")
    ax.set_title("Q-Network Training Loss"); ax.grid(alpha=0.3)

    # Action distribution pie
    action_totals = stats["action_totals"]
    names  = ["scale_up", "scale_down", "migrate_aws", "migrate_azure"]
    values = [action_totals[i] for i in range(4)]
    if sum(values) > 0:
        axins = fig.add_axes([0.68, 0.12, 0.12, 0.30])
        axins.pie(values, labels=names, autopct="%d%%",
                  textprops={"fontsize": 6},
                  colors=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"])
        axins.set_title("Actions", fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Training plot saved → {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    # ── Load XGBoost ──────────────────────────────────────────────────────────
    try:
        xgb_model = load_xgb_model()
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        return 1

    # ── Build cost history buffer ─────────────────────────────────────────────
    logger.info("Building cost history buffer (2000 hours)...")
    cost_history = build_cost_history(n=2000)
    logger.info(f"  Cost history: min=${cost_history.min():.3f}  "
                f"max=${cost_history.max():.3f}  "
                f"mean=${cost_history.mean():.3f}")

    # ── Create environment and agent ──────────────────────────────────────────
    env   = ForecastDrivenCloudEnvironment(
        xgb_model      = xgb_model,
        cost_history   = cost_history,
        episode_length = EPISODE_LENGTH,
    )
    agent = DQNAgent(
        state_dim     = ForecastDrivenCloudEnvironment.STATE_DIM,
        action_dim    = ForecastDrivenCloudEnvironment.ACTION_DIM,
        hidden_dim    = 256,
        lr            = 0.0005,
        gamma         = 0.99,
        epsilon_start = 1.0,
        epsilon_end   = 0.05,
        epsilon_decay = 0.997,
        device        = DEVICE,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    stats = train(agent, env)

    # ── Save ──────────────────────────────────────────────────────────────────
    agent.save(MODELS_DIR / "rl_agent_integrated.pth")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_training(stats, PLOTS_DIR / "rl_training_integrated.png")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    evaluate(agent, env)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 62)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 62)
    logger.info(f"  Final epsilon   : {agent.epsilon:.4f}")
    logger.info(f"  Success rate    : "
                f"{stats['success_count']/NUM_EPISODES*100:.1f}%")
    logger.info(f"  Avg reward (last 50): "
                f"{np.mean(stats['rewards'][-50:]):.3f}")
    logger.info(f"  Avg cost/hr (last 50): "
                f"${np.mean(stats['costs'][-50:]):.3f}")
    logger.info(f"  Model → {MODELS_DIR / 'rl_agent_integrated.pth'}")
    logger.info(f"  Plot  → {PLOTS_DIR / 'rl_training_integrated.png'}")
    logger.info("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())