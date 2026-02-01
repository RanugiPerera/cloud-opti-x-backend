import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class DQNetwork(nn.Module):
    """
    Deep Q-Network for cloud resource optimization

    Takes current state (workload, costs, etc.) and outputs Q-values for each action
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

        # initialize weights
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, state):
        """
        Forward pass

        Args:
            state: (batch, state_dim) - current environment state

        Returns:
            q_values: (batch, action_dim) - Q-value for each action
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)

        return q_values


class ReplayBuffer:
    """
    Experience replay buffer for DQN
    Stores past experiences and samples random batches for training
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store experience"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample random batch of experiences"""
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent for cloud cost optimization

    State: [cpu_usage, memory_usage, storage_gb, network_gb, current_cost, provider_id]
    Actions: [scale_up, scale_down, migrate_aws, migrate_azure]
    Reward: cost_savings - performance_penalty
    """

    def __init__(
            self,
            state_dim=6,
            action_dim=4,
            hidden_dim=128,
            lr=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            device='cpu'
    ):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.gamma = gamma

        # epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-networks
        self.q_network = DQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = DQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

        # replay buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)

        # action names for interpretability
        self.action_names = ['scale_up', 'scale_down', 'migrate_aws', 'migrate_azure']

    def select_action(self, state, greedy=False):
        """
        Select action using epsilon-greedy policy

        Args:
            state: current state (numpy array)
            greedy: if True, always select best action (for evaluation)

        Returns:
            action: integer action index
        """
        # exploration: random action
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        # exploitation: best action according to Q-network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def update(self, batch_size=32):
        """
        Update Q-network using experience replay
        """
        if len(self.replay_buffer) < batch_size:
            return None

        # sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path):
        """Save model weights"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
        print(f"RL agent saved to {path}")

    def load(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"RL agent loaded from {path}")

    def get_action_name(self, action):
        """Get human-readable action name"""
        return self.action_names[action]


class CloudEnvironment:
    """
    Simulated cloud environment for RL training
    Simulates costs, performance, and state transitions
    """

    def __init__(self):
        self.state = None
        self.reset()

    def reset(self):
        """Reset to initial state"""
        # state: [cpu, memory, storage_gb, network_gb, cost, provider]
        self.state = np.array([
            np.random.uniform(0.3, 0.7),  # cpu usage (0-1)
            np.random.uniform(0.3, 0.7),  # memory usage (0-1)
            np.random.uniform(50, 300),  # storage GB
            np.random.uniform(1, 10),  # network GB per hour
            np.random.uniform(0.5, 2.0),  # current cost per hour
            0.0 if np.random.random() < 0.5 else 1.0  # provider: 0=aws, 1=azure
        ])
        return self.state

    def step(self, action):
        """
        Take action and return next state, reward, done

        Actions:
        0 - scale_up
        1 - scale_down
        2 - migrate_aws
        3 - migrate_azure
        """
        # unpack state
        cpu, memory, storage, network, cost, provider = self.state

        # simulate action effects
        if action == 0:  # scale_up
            cpu = min(cpu + 0.1, 0.95)
            memory = min(memory + 0.1, 0.95)
            storage = min(storage + 50, 500)
            cost = cost * 1.25  # cost increases
            performance_gain = 0.15

        elif action == 1:  # scale_down
            cpu = max(cpu - 0.1, 0.1)
            memory = max(memory - 0.1, 0.1)
            storage = max(storage - 50, 10)
            cost = cost * 0.75  # cost decreases
            performance_gain = -0.05  # slight performance loss

        elif action == 2:  # migrate to aws
            if provider != 0.0:
                # aws has cheaper compute but slightly more network costs
                cost = cost * 0.92
                network = network * 1.05
                provider = 0.0
                performance_gain = 0.05
            else:
                performance_gain = 0  # already on aws

        elif action == 3:  # migrate to azure
            if provider != 1.0:
                # azure slightly more expensive but better performance
                cost = cost * 1.08
                network = network * 0.98
                provider = 1.0
                performance_gain = 0.10
            else:
                performance_gain = 0  # already on azure

        # add realistic noise
        cpu += np.random.normal(0, 0.02)
        memory += np.random.normal(0, 0.02)
        storage += np.random.normal(0, 5)
        network += np.random.normal(0, 0.5)
        cost += np.random.normal(0, 0.01)

        # clip values to realistic ranges
        cpu = np.clip(cpu, 0.05, 0.95)
        memory = np.clip(memory, 0.05, 0.95)
        storage = np.clip(storage, 10, 500)
        network = np.clip(network, 0.5, 20)
        cost = np.clip(cost, 0.1, 5.0)

        # calculate reward: minimize cost while maintaining performance
        # good state: low cost + good performance
        cost_penalty = cost  # higher cost = worse
        performance_reward = performance_gain * 2  # reward good performance

        # bonus for efficient resource usage
        efficiency_bonus = 0
        if cpu < 0.6 and cost < 1.0:
            efficiency_bonus = 0.5

        reward = -cost_penalty + performance_reward + efficiency_bonus

        # check if done (reached optimal state or bad state)
        done = False
        if cost < 0.3 and cpu < 0.6:  # good efficient state
            done = True
            reward += 5.0  # bonus for reaching good state
        elif cost > 4.0:  # too expensive
            done = True
            reward -= 3.0  # penalty

        # update state
        self.state = np.array([cpu, memory, storage, network, cost, provider])

        return self.state, reward, done