import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

import numpy as np
import matplotlib.pyplot as plt
from models.rl_agent import DQNAgent, CloudEnvironment
from utils.config import Config

# config
MODELS_DIR = Config.MODELS_DIR
MODELS_DIR.mkdir(exist_ok=True)

# hyperparameters
NUM_EPISODES = 500
MAX_STEPS = 100
BATCH_SIZE = 32
UPDATE_TARGET_EVERY = 10

DEVICE = 'cpu'

print("=" * 60)
print("RL Agent Training - Cloud Cost Optimization")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Episodes: {NUM_EPISODES}")
print(f"Max steps per episode: {MAX_STEPS}")
print("=" * 60)

# create environment and agent
env = CloudEnvironment()
agent = DQNAgent(
    state_dim=6,
    action_dim=4,
    hidden_dim=128,
    lr=0.001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    device=DEVICE
)

# training loop
episode_rewards = []
episode_losses = []
episode_costs = []  # track final costs
episode_steps = []  # track steps to completion
success_count = 0  # count successful episodes

print("\nStarting training...")

for episode in range(NUM_EPISODES):
    state = env.reset()
    episode_reward = 0
    episode_loss = []
    final_cost = 0

    for step in range(MAX_STEPS):
        # select action
        action = agent.select_action(state)

        # take action
        next_state, reward, done = env.step(action)

        # store experience
        agent.replay_buffer.push(state, action, reward, next_state, done)

        # update agent
        loss = agent.update(BATCH_SIZE)
        if loss is not None:
            episode_loss.append(loss)

        # accumulate reward
        episode_reward += reward

        # track final cost (state index 4 is cost)
        final_cost = next_state[4]

        # move to next state
        state = next_state

        if done:
            # count as success if cost is low
            if final_cost < 0.5:
                success_count += 1
            break

    # update target network
    if episode % UPDATE_TARGET_EVERY == 0:
        agent.update_target_network()

    # decay epsilon
    agent.decay_epsilon()

    # record stats
    episode_rewards.append(episode_reward)
    episode_costs.append(final_cost)
    episode_steps.append(step + 1)
    avg_loss = np.mean(episode_loss) if episode_loss else 0
    episode_losses.append(avg_loss)

    # print progress
    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        avg_cost = np.mean(episode_costs[-50:])
        success_rate = success_count / (episode + 1) * 100

        print(f"Episode {episode + 1}/{NUM_EPISODES} - "
              f"Avg Reward: {avg_reward:.3f}, "
              f"Avg Cost: ${avg_cost:.3f}, "
              f"Success Rate: {success_rate:.1f}%, "
              f"Epsilon: {agent.epsilon:.3f}, "
              f"Loss: {avg_loss:.4f}")

print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)

# save model
model_path = MODELS_DIR / 'rl_agent.pth'
agent.save(model_path)

# plot training curves with 3 subplots
fig = plt.figure(figsize=(18, 5))

# rewards
ax1 = plt.subplot(1, 3, 1)
ax1.plot(episode_rewards, alpha=0.3, label='Episode Reward', color='blue')
# moving average
window = 20
moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
ax1.plot(range(window - 1, len(episode_rewards)), moving_avg,
         label=f'{window}-Episode Moving Avg', linewidth=2, color='darkblue')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.set_title('Training Rewards')
ax1.legend()
ax1.grid(True, alpha=0.3)

# costs
ax2 = plt.subplot(1, 3, 2)
ax2.plot(episode_costs, alpha=0.3, label='Final Cost', color='red')
moving_avg_cost = np.convolve(episode_costs, np.ones(window) / window, mode='valid')
ax2.plot(range(window - 1, len(episode_costs)), moving_avg_cost,
         label=f'{window}-Episode Moving Avg', linewidth=2, color='darkred')
ax2.set_xlabel('Episode')
ax2.set_ylabel('Cost ($)')
ax2.set_title('Episode Final Costs')
ax2.legend()
ax2.grid(True, alpha=0.3)

# losses
ax3 = plt.subplot(1, 3, 3)
ax3.plot(episode_losses, alpha=0.5, color='green')
ax3.set_xlabel('Episode')
ax3.set_ylabel('Loss')
ax3.set_title('Training Loss')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(MODELS_DIR / 'rl_training_curve.png', dpi=150)
print(f"\nTraining plot saved to: {MODELS_DIR / 'rl_training_curve.png'}")

# detailed testing
print("\n" + "=" * 60)
print("Testing trained agent (Detailed Evaluation)...")
print("=" * 60)

test_episodes = 20
test_rewards = []
test_costs = []
test_steps = []
test_success = 0
action_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # count each action type

for ep in range(test_episodes):
    state = env.reset()
    episode_reward = 0
    actions_taken = []

    for step in range(50):
        action = agent.select_action(state, greedy=True)  # greedy policy
        action_counts[action] += 1
        actions_taken.append(agent.get_action_name(action))
        next_state, reward, done = env.step(action)
        episode_reward += reward
        state = next_state

        if done:
            break

    final_cost = state[4]
    test_rewards.append(episode_reward)
    test_costs.append(final_cost)
    test_steps.append(step + 1)

    if final_cost < 0.5:
        test_success += 1

    print(f"Test Ep {ep + 1}: Reward={episode_reward:.3f}, Cost=${final_cost:.3f}, Steps={step + 1}")

# calculate metrics
print("\n" + "=" * 60)
print("EVALUATION METRICS")
print("=" * 60)
print(f"Average Test Reward: {np.mean(test_rewards):.3f} ± {np.std(test_rewards):.3f}")
print(f"Average Final Cost: ${np.mean(test_costs):.3f} ± ${np.std(test_costs):.3f}")
print(f"Average Steps: {np.mean(test_steps):.1f} ± {np.std(test_steps):.1f}")
print(f"Success Rate: {test_success / test_episodes * 100:.1f}% ({test_success}/{test_episodes})")
print(f"Cost Reduction: {(2.0 - np.mean(test_costs)) / 2.0 * 100:.1f}% (vs baseline $2.00)")

print(f"\nAction Distribution:")
total_actions = sum(action_counts.values())
for action_id, count in action_counts.items():
    action_name = agent.get_action_name(action_id)
    percentage = count / total_actions * 100
    print(f"  {action_name}: {count} times ({percentage:.1f}%)")

# training summary
print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
print(f"Total Episodes: {NUM_EPISODES}")
print(f"Final Epsilon: {agent.epsilon:.4f}")
print(f"Training Success Rate: {success_count / NUM_EPISODES * 100:.1f}%")
print(f"Average Reward (last 50): {np.mean(episode_rewards[-50:]):.3f}")
print(f"Average Cost (last 50): ${np.mean(episode_costs[-50:]):.3f}")
print(f"Model saved to: {model_path}")

print("\n✅ RL training complete!")