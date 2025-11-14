import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import argparse
# pip install swig "gymnasium[box2d]" pygame numpy matplotlib tqdm torch

class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def size(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q-Network for DQN"""

    def __init__(self, num_states, num_actions, hidden_size=256):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(num_states, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class DQN:
    def __init__(self, path=None, render_mode=None, device='cpu'):
        # Use discrete LunarLander for DQN
        self.env = gym.make('LunarLander-v3', render_mode=render_mode, continuous=False)

        # Device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.gamma = 0.99
        self.max_steps_per_episode = 1000

        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        # DQN Hyperparameters
        self.batch_size = 128
        self.lr = 0.0001
        self.hidden_size = 256
        self.buffer_capacity = 100000
        self.min_buffer_size = 10000  # Start training after this many experiences
        self.target_update_frequency = 1000  # Update target network every N steps
        self.train_frequency = 4  # Train every N steps

        # Epsilon-greedy parameters
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start

        # Initialize networks
        self.q_network = QNetwork(self.num_states, self.num_actions, self.hidden_size).to(self.device)
        self.target_network = QNetwork(self.num_states, self.num_actions, self.hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_capacity)

        if path is not None:
            self.q_network.load_state_dict(torch.load(path, map_location=self.device))
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"Loaded model from {path}")

    def get_action(self, state, deterministic=False):
        """Select action using epsilon-greedy policy"""
        if not deterministic and random.random() < self.epsilon:
            # Random action (exploration)
            return self.env.action_space.sample()
        else:
            # Greedy action (exploitation)
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            return q_values.argmax().item()

    def update(self):
        """Update Q-network using a batch from replay buffer"""
        if self.replay_buffer.size() < self.batch_size:
            return 0.0

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self):
        episode_rewards = []
        running_reward = 0
        episode_count = 0
        total_steps = 0
        total_loss = 0
        loss_count = 0

        average = deque(maxlen=100)

        print("Starting DQN training on LunarLander-v3...")
        print(f"Hyperparameters: batch_size={self.batch_size}, lr={self.lr}, "
              f"buffer_capacity={self.buffer_capacity}")
        print(f"Epsilon: start={self.epsilon_start}, end={self.epsilon_end}, decay={self.epsilon_decay}")

        state, info = self.env.reset()
        episode_reward = 0

        while True:
            # Render environment if enabled
            if self.env.render_mode == 'human':
                self.env.render()

            # Select action
            action = self.get_action(state)

            # Execute action
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Store transition
            self.replay_buffer.add(state, action, reward, next_state, done)

            episode_reward += reward
            state = next_state
            total_steps += 1

            # Train network
            if total_steps % self.train_frequency == 0 and self.replay_buffer.size() >= self.min_buffer_size:
                loss = self.update()
                total_loss += loss
                loss_count += 1

            # Update target network
            if total_steps % self.target_update_frequency == 0:
                self.update_target_network()
                if loss_count > 0:
                    avg_loss = total_loss / loss_count
                    print(f"Step {total_steps}: Target network updated, Avg Loss: {avg_loss:.4f}, "
                          f"Epsilon: {self.epsilon:.4f}, Buffer size: {self.replay_buffer.size()}")
                    total_loss = 0
                    loss_count = 0

            if done:
                episode_rewards.append(episode_reward)
                average.append(episode_reward)
                running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
                episode_count += 1

                # Decay epsilon
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

                # Reset environment
                state, info = self.env.reset()
                episode_reward = 0

                # Log progress
                if episode_count % 10 == 0:
                    num_avg_episodes = min(len(average), 100)
                    avg_reward = np.mean(list(average)[-100:]) if len(average) > 0 else 0
                    print(f"Episode: {episode_count}, Running Reward: {running_reward:.2f}, "
                          f"Avg (last {num_avg_episodes} ep): {avg_reward:.2f}, "
                          f"Epsilon: {self.epsilon:.4f}, Total Steps: {total_steps}")

                # Save model every 100 episodes
                if episode_count == 10:
                    save_path = f'DQN_LunarLander_ep{episode_count}.pth'
                    torch.save(self.q_network.state_dict(), save_path)
                    print(f"Model saved: {save_path}")
                if episode_count % 100 == 0:
                    save_path = f'DQN_LunarLander_ep{episode_count}.pth'
                    torch.save(self.q_network.state_dict(), save_path)
                    print(f"Model saved: {save_path}")

                # Check if solved
                if len(average) >= 100:
                    avg_reward = np.mean(list(average))
                    if avg_reward > 200:
                        print(f"Solved at episode {episode_count}! Average reward: {avg_reward:.2f}")
                        torch.save(self.q_network.state_dict(), 'DQN_LunarLander_final.pth')
                        return self.q_network

    def test(self, path, num_episodes=10):
        """Test trained policy"""
        if path:
            self.q_network.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Loaded model from {path}")

        total_rewards = []

        for episode in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0

            for step in range(self.max_steps_per_episode):
                if self.env.render_mode == 'human':
                    self.env.render()
                action = self.get_action(state, deterministic=True)
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

                if done:
                    break

            total_rewards.append(episode_reward)
            print(f'Episode {episode + 1}: {episode_reward:.2f}')

        avg_reward = np.mean(total_rewards)
        print(f'Average reward over {num_episodes} episodes: {avg_reward:.2f}')
        return avg_reward

    def test_all_models(self, num_episodes=5):
        """Test all saved models and display their performance"""
        import os
        import glob

        # Find all saved model files
        model_files = glob.glob('DQN_LunarLander_ep*.pth')

        if not model_files:
            print("No saved models found!")
            return

        # Sort by episode number
        def get_episode_num(filename):
            import re
            match = re.search(r'ep(\d+)', filename)
            return int(match.group(1)) if match else 0

        model_files.sort(key=get_episode_num)

        print(f"\nFound {len(model_files)} saved models")
        print("=" * 70)

        results = []

        for model_file in model_files:
            episode_num = get_episode_num(model_file)
            print(f"\nTesting model: {model_file} (Episode {episode_num})")
            print("-" * 70)

            # Load and test the model
            self.q_network.load_state_dict(torch.load(model_file, map_location=self.device))
            avg_reward = self.test(None, num_episodes=num_episodes)

            results.append({
                'model': model_file,
                'episode': episode_num,
                'avg_reward': avg_reward
            })

            print("-" * 70)

        # Display summary
        print("\n" + "=" * 70)
        print("SUMMARY OF ALL MODELS")
        print("=" * 70)
        print(f"{'Model':<40} {'Episode':<10} {'Avg Reward':<15}")
        print("-" * 70)

        for result in results:
            print(f"{result['model']:<40} {result['episode']:<10} {result['avg_reward']:<15.2f}")

        # Find best model
        best_model = max(results, key=lambda x: x['avg_reward'])
        print("=" * 70)
        print(f"Best Model: {best_model['model']} with avg reward: {best_model['avg_reward']:.2f}")
        print("=" * 70)

        return results


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str,
                        help="train - train model, test - test single model, test_all - test all saved models",
                        default='train')
    parser.add_argument("--path", type=str, help="policy path for single test", default=None)
    parser.add_argument("--episodes", type=int, help="number of episodes for testing", default=5)
    parser.add_argument("--render", action='store_true', help="enable rendering/visualization")
    parser.add_argument("--no-render", dest='render', action='store_false', help="disable rendering/visualization")
    parser.add_argument("--device", type=str, help="cpu or cuda", default='cpu')
    parser.set_defaults(render=True)
    return parser


if __name__ == '__main__':
    args = get_args().parse_args()
    # Enable rendering based on command line argument
    render_mode = 'human' if args.render else None

    if args.render:
        print("Rendering enabled: You will see the environment visualization")
    else:
        print("Rendering disabled: Training/testing will run faster without visualization")

    dqn = DQN(args.path if args.mode == 'train' else None, render_mode=render_mode, device=args.device)

    if args.mode == 'train':
        dqn.train()
    elif args.mode == 'test':
        if args.path is None:
            print("Error: Please provide --path argument for testing a single model")
        else:
            dqn.test(args.path, num_episodes=args.episodes)
    elif args.mode == 'test_all':
        dqn.test_all_models(num_episodes=args.episodes)
    else:
        print(f"Unknown mode: {args.mode}. Use 'train', 'test', or 'test_all'")