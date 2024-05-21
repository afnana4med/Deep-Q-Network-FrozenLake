import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import numpy as np
import matplotlib.pyplot as plt

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))

class DeepQNetwork(nn.Module):
    def __init__(self, channels, actions):
        # Network layers and forward pass implementation
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)  # 64 channels, 4x4 grid after pooling
        self.fc2 = nn.Linear(64, actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    # Add experiences to the buffer
    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, size):
        return random.sample(self.buffer, size)
    
    # Randomly sample experiences from the buffer
    def __len__(self):
        return len(self.buffer)

class FrozenLakeAgent:
    def __init__(self, is_slippery=None):
        # Initializes the environment, neural networks, and other parameters
        self.is_slippery = is_slippery
        self.env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=self.is_slippery, render_mode='rgb_array')
        self.replay_memory = ExperienceReplay(1000)
        self.policy_net = DeepQNetwork(3, self.env.action_space.n)
        self.target_net = DeepQNetwork(3, self.env.action_space.n)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.discount_factor = 0.99
        self.epsilon_decay = np.linspace(1.0, 0.01, 2000)  # Adjusted to match number of episodes
        self.ACTIONS = ['L', 'D', 'R', 'U']

    def configure_environment(self, is_slippery=None):
        if is_slippery is not None:
            self.is_slippery = is_slippery
        self.env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=self.is_slippery, render_mode='rgb_array')

    def select_action(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.env.action_space.n)]], dtype=torch.long)

    def train(self, num_episodes, is_slippery=None):
        # Main loop for training the agent
        self.configure_environment(is_slippery)

        rewards_history = []

        # Print initial random policy
        self.print_policy(before=True)

        for episode in range(num_episodes):
            state = self.env.reset()[0]
            state = self.state_to_tensor(state)
            terminated = False
            total_reward = 0
            step_count = 0

            while not terminated:
                epsilon = self.epsilon_decay[min(episode, len(self.epsilon_decay)-1)]
                action = self.select_action(state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                next_state = self.state_to_tensor(next_state)
                reward = max(min(reward, 1.0), -1.0)
                total_reward += reward

                self.replay_memory.add(Experience(state, action, next_state, torch.tensor([reward]), terminated))
                state = next_state
                self.optimize_model(32)
                step_count += 1
                if terminated or truncated or step_count >= 100:
                    break

            rewards_history.append(total_reward)
            if episode % 10 == 0:
                self.update_target_network()

            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")

        # Print trained policy
        self.print_policy(before=False)

        return rewards_history

    def optimize_model(self, batch_size):
        # Function to perform one step of training on the sampled batch
        if len(self.replay_memory) < batch_size:
            return
        transitions = self.replay_memory.sample(batch_size)
        batch = Experience(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).view(-1, 1)
        reward_batch = torch.cat(batch.reward)
        non_final_mask = torch.tensor([not done for done in batch.done], dtype=torch.bool)
        non_final_next_states = torch.cat([s for s, d in zip(batch.next_state, batch.done) if not d])

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=state_action_values.device)
        if non_final_mask.any():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target_network(self):
        # Synchronize target network with policy network periodically
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def state_to_tensor(self, state):
        # Converts environment state into tensor for neural network processing
        tensor = torch.zeros(1, 3, 4, 4)
        row, col = divmod(state, 4)
        tensor[0, :, row, col] = 1.0
        return tensor

    def print_policy(self, before=True):
        # Output the learned policy
        description = "random, before training" if before else "trained"
        print(f"Policy ({description}):")
        for state in range(16):
            state_tensor = self.state_to_tensor(state)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor).squeeze().tolist()
            best_action = self.ACTIONS[q_values.index(max(q_values))]
            formatted_q_values = ', '.join([f"{value:+.2f}" for value in q_values])
            print(f"{state:02},{best_action},[{formatted_q_values}]", end=' ')
            if (state + 1) % 4 == 0:
                print()

    def test(self, episodes, is_slippery=None):
        # Test the trained agent in the environment
        self.configure_environment(is_slippery)
        self.env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=self.is_slippery, render_mode='human')
        for episode in range(episodes):
            state = self.env.reset()[0]
            terminated = False
            step_count = 0
            total_reward = 0

            while not terminated:
                state_tensor = self.state_to_tensor(state)
                with torch.no_grad():
                    action = self.policy_net(state_tensor).max(1)[1].item()
                state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                step_count += 1

                if terminated or truncated or step_count >= 100:
                    break

            print(f"Test Episode {episode}, Total Reward: {total_reward}")

        self.env.close()

if __name__ == '__main__':
    agent = FrozenLakeAgent()
    rewards_history_slippery = agent.train(2000, is_slippery=True)
    agent.test(10, is_slippery=True)
    rewards_history_non_slippery = agent.train(2000, is_slippery=False)
    agent.test(10, is_slippery=False)
    
    # Code for plotting results

    # Plot Rewards History for both slippery and non-slippery environments
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history_slippery, label='Slippery')
    plt.plot(rewards_history_non_slippery, label='Non-Slippery')
    plt.title('Rewards History Comparison')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.show()

    # Plot Moving Average of Rewards for non-slippery environment
    def moving_average(data, window_size=50):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    ma_rewards_non_slippery = moving_average(rewards_history_non_slippery, 50)
    ma_rewards_slippery = moving_average(rewards_history_slippery, 50)

    plt.figure(figsize=(12, 6))
    plt.plot(ma_rewards_non_slippery, label='Non-Slippery')
    plt.plot(ma_rewards_slippery, label='Slippery')
    plt.title('Moving Average of Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.show()

    # Plot Epsilon Decay
    plt.figure(figsize=(12, 6))
    plt.plot(agent.epsilon_decay)
    plt.title('Epsilon Decay Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.show()