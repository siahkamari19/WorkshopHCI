import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import argparse


# pip install "gymnasium[box2d]" pygame numpy matplotlib tqdm torch


class ReplayBuffer:
    """Experience replay buffer for SAC"""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])

        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )

    def size(self):
        return len(self.buffer)


class CNNFeatureExtractor(nn.Module):
    """CNN for extracting features from image observations"""

    def __init__(self, output_size=256):
        super(CNNFeatureExtractor, self).__init__()

        # Input: 96x96x3 (CarRacing image)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)  # -> 23x23x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # -> 10x10x64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # -> 8x8x64

        # Calculate the flattened size: 8*8*64 = 4096
        self.fc = nn.Linear(8 * 8 * 64, output_size)

    def forward(self, x):
        # Normalize pixel values to [0, 1]
        x = x / 255.0

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.reshape(x.size(0), -1)  # Flatten
        x = F.relu(self.fc(x))

        return x


class Actor(nn.Module):
    """Actor network that outputs mean and log_std for Gaussian policy"""

    def __init__(self, num_actions, hidden_size=256, feature_size=256):
        super(Actor, self).__init__()

        self.feature_extractor = CNNFeatureExtractor(feature_size)

        self.fc1 = nn.Linear(feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, num_actions)
        self.log_std = nn.Linear(hidden_size, num_actions)

        # Action rescaling for CarRacing: steering [-1, 1], gas [0, 1], brake [0, 1]
        self.action_scale = torch.FloatTensor([1.0, 1.0, 1.0])
        self.action_bias = torch.FloatTensor([0.0, 0.0, 0.0])

    def forward(self, state):
        x = self.feature_extractor(state)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)

        action_scale = self.action_scale.to(state.device)
        action_bias = self.action_bias.to(state.device)

        action = y_t * action_scale + action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing action bounds
        log_prob -= torch.log(action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * action_scale + action_bias

        return action, log_prob, mean


class Critic(nn.Module):
    """Critic network (Q-function)"""

    def __init__(self, num_actions, hidden_size=256, feature_size=256):
        super(Critic, self).__init__()

        self.feature_extractor = CNNFeatureExtractor(feature_size)

        # Q1 architecture
        self.fc1 = nn.Linear(feature_size + num_actions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = self.feature_extractor(state)
        x = torch.cat([x, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SAC:
    def __init__(self, path=None, render_mode=None, device='cpu'):
        # Use CarRacing environment
        self.env = gym.make('CarRacing-v3', render_mode=render_mode, continuous=True)

        # Device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.gamma = 0.99
        self.tau = 0.005
        self.max_steps_per_episode = 1000

        # CarRacing has image observations (96x96x3) and 3 continuous actions
        self.num_actions = self.env.action_space.shape[0]  # 3: steering, gas, brake

        # Hyperparameters
        self.batch_size = 64  # Reduced for memory efficiency with images
        self.buffer_capacity = 50000  # Reduced for memory efficiency
        self.actor_lr = 0.0001
        self.critic_lr = 0.0003
        self.alpha_lr = 0.0001
        self.hidden_size = 256
        self.feature_size = 256
        self.update_frequency = 1

        # Initialize networks
        self.actor = Actor(self.num_actions, self.hidden_size, self.feature_size).to(self.device)
        self.critic_1 = Critic(self.num_actions, self.hidden_size, self.feature_size).to(self.device)
        self.critic_2 = Critic(self.num_actions, self.hidden_size, self.feature_size).to(self.device)
        self.target_critic_1 = Critic(self.num_actions, self.hidden_size, self.feature_size).to(self.device)
        self.target_critic_2 = Critic(self.num_actions, self.hidden_size, self.feature_size).to(self.device)

        # Copy weights to target networks
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.critic_lr)

        # Auto-tune temperature parameter
        self.target_entropy = -self.num_actions
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_capacity)

        if path is not None:
            self.actor.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Loaded model from {path}")

    def preprocess_state(self, state):
        """Preprocess the image state"""
        # state is (96, 96, 3) - convert to (3, 96, 96) for PyTorch
        state = np.transpose(state, (2, 0, 1))
        return state

    def get_action(self, state, deterministic=False):
        """Sample action from policy"""
        state = self.preprocess_state(state)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if deterministic:
            _, _, action = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)

        return action.detach().cpu().numpy()[0]

    def update(self, states, actions, rewards, next_states, dones):
        """Update SAC networks"""
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current alpha value
        alpha = self.log_alpha.exp()

        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_q1 = self.target_critic_1(next_states, next_actions)
            target_q2 = self.target_critic_2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * (target_q - alpha * next_log_probs)

        # Current Q-values
        current_q1 = self.critic_1(states, actions)
        current_q2 = self.critic_2(states, actions)

        # Critic losses
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        # Update critic 1
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        # Update critic 2
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Update actor
        new_actions, log_probs, _ = self.actor.sample(states)
        q1 = self.critic_1(states, new_actions)
        q2 = self.critic_2(states, new_actions)
        q = torch.min(q1, q2)

        actor_loss = (alpha * log_probs - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update temperature parameter (alpha)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update target networks
        self._soft_update(self.target_critic_1, self.critic_1)
        self._soft_update(self.target_critic_2, self.critic_2)

        return critic_1_loss.item(), critic_2_loss.item(), actor_loss.item(), alpha.item()

    def _soft_update(self, target_model, source_model):
        """Soft update target network"""
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    def train(self):
        episode_rewards = []
        running_reward = 0
        episode_count = 0
        total_steps = 0

        average = deque(maxlen=100)

        print("Starting SAC training on CarRacing-v3...")
        print(f"Hyperparameters: batch_size={self.batch_size}, actor_lr={self.actor_lr}, critic_lr={self.critic_lr}")

        while True:
            state, info = self.env.reset()
            episode_reward = 0
            negative_reward_count = 0

            for step in range(self.max_steps_per_episode):
                # Render environment if enabled
                if self.env.render_mode == 'human':
                    self.env.render()

                # Select action
                if total_steps < 5000:  # Random exploration at start
                    action = self.env.action_space.sample()
                else:
                    action = self.get_action(state)

                # Execute action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # CarRacing gives negative rewards when off track
                # Track consecutive negative rewards to end episode early
                if reward < 0:
                    negative_reward_count += 1
                else:
                    negative_reward_count = 0

                # End episode if stuck off track
                if negative_reward_count > 100:
                    done = True

                # Store transition (preprocess states for buffer)
                state_processed = self.preprocess_state(state)
                next_state_processed = self.preprocess_state(next_state)
                self.replay_buffer.add(state_processed, action, reward, next_state_processed, done)

                episode_reward += reward
                state = next_state
                total_steps += 1

                # Train after collecting enough samples
                if self.replay_buffer.size() >= self.batch_size and total_steps % self.update_frequency == 0:
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = \
                        self.replay_buffer.sample(self.batch_size)

                    critic_1_loss, critic_2_loss, actor_loss, alpha = self.update(
                        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
                    )

                if done:
                    break

            # Update statistics
            episode_rewards.append(episode_reward)
            average.append(episode_reward)
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            episode_count += 1

            # Log progress
            if episode_count % 10 == 0:
                num_avg_episodes = min(len(average), 100)
                avg_reward = np.mean(list(average)[-100:]) if len(average) > 0 else episode_reward
                current_alpha = self.log_alpha.exp().item()
                print(f"Episode: {episode_count}, Running Reward: {running_reward:.2f}, "
                      f"Avg (last {num_avg_episodes} ep): {avg_reward:.2f}, Alpha: {current_alpha:.3f}, Total Steps: {total_steps}")

            # Save model every 100 episodes
            if episode_count == 10:
                save_path = f'SAC_CarRacing_ep{episode_count}.pth'
                torch.save(self.actor.state_dict(), save_path)
                print(f"Model saved: {save_path}")

            if episode_count % 100 == 0:
                save_path = f'SAC_CarRacing_ep{episode_count}.pth'
                torch.save(self.actor.state_dict(), save_path)
                print(f"Model saved: {save_path}")

            # Check if solved (CarRacing is considered solved at 900+ average reward)
            if len(average) >= 100:
                avg_reward = np.mean(list(average))
                if avg_reward > 900:
                    print(f"Solved at episode {episode_count}! Average reward: {avg_reward:.2f}")
                    torch.save(self.actor.state_dict(), 'SAC_CarRacing_final.pth')
                    return self.actor

    def test(self, path, num_episodes=10):
        """Test trained policy"""
        if path:
            self.actor.load_state_dict(torch.load(path, map_location=self.device))
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
        model_files = glob.glob('SAC_CarRacing_ep*.pth')

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
            self.actor.load_state_dict(torch.load(model_file, map_location=self.device))
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

    sac = SAC(args.path if args.mode == 'train' else None, render_mode=render_mode, device=args.device)

    if args.mode == 'train':
        sac.train()
    elif args.mode == 'test':
        if args.path is None:
            print("Error: Please provide --path argument for testing a single model")
        else:
            sac.test(args.path, num_episodes=args.episodes)
    elif args.mode == 'test_all':
        sac.test_all_models(num_episodes=args.episodes)
    else:
        print(f"Unknown mode: {args.mode}. Use 'train', 'test', or 'test_all'")