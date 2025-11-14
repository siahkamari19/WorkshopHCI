import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque
import argparse
# pip install swig "gymnasium[box2d]" pygame numpy matplotlib tqdm torch

class RolloutBuffer:
    """Buffer for storing trajectories for PPO"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def get(self):
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.FloatTensor(np.array(self.actions)),
            torch.FloatTensor(np.array(self.rewards)),
            torch.FloatTensor(np.array(self.dones)),
            torch.FloatTensor(np.array(self.log_probs)),
            torch.FloatTensor(np.array(self.values))
        )

    def size(self):
        return len(self.states)


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""

    def __init__(self, num_states, num_actions, hidden_size=256):
        super(ActorCritic, self).__init__()

        # Shared feature extraction
        self.fc1 = nn.Linear(num_states, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Actor head
        self.mean = nn.Linear(hidden_size, num_actions)
        self.log_std = nn.Parameter(torch.zeros(1, num_actions))

        # Critic head
        self.value = nn.Linear(hidden_size, 1)

        # Action rescaling
        self.action_scale = 1.0
        self.action_bias = 0.0

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Actor outputs
        mean = self.mean(x)
        std = self.log_std.exp().expand_as(mean)

        # Critic output
        value = self.value(x)

        return mean, std, value

    def get_action(self, state, deterministic=False):
        """Sample action from policy"""
        mean, std, value = self.forward(state)

        if deterministic:
            action = torch.tanh(mean)
        else:
            normal = Normal(mean, std)
            x_t = normal.sample()
            action = torch.tanh(x_t)
            log_prob = normal.log_prob(x_t)

            # Apply tanh correction
            log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + 1e-6)
            log_prob = log_prob.sum(-1, keepdim=True)

        action = action * self.action_scale + self.action_bias

        if deterministic:
            return action, None, value
        else:
            return action, log_prob, value

    def evaluate_actions(self, states, actions):
        """Evaluate actions for PPO update"""
        mean, std, value = self.forward(states)

        # Convert actions back to pre-tanh space
        actions_normalized = (actions - self.action_bias) / self.action_scale
        actions_normalized = torch.clamp(actions_normalized, -0.999, 0.999)
        x_t = torch.atanh(actions_normalized)

        normal = Normal(mean, std)
        log_prob = normal.log_prob(x_t)

        # Apply tanh correction
        log_prob -= torch.log(self.action_scale * (1 - actions_normalized.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        entropy = normal.entropy().sum(-1, keepdim=True)

        return log_prob, value, entropy


class PPO:
    def __init__(self, path=None, render_mode=None, device='cpu'):
        # Use continuous LunarLander for PPO
        self.env = gym.make('LunarLander-v3', render_mode=render_mode, continuous=True)

        # Device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.max_steps_per_episode = 1000

        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]

        # PPO Hyperparameters
        self.batch_size = 64
        self.n_epochs = 4  # Reduced from 10 to prevent overfitting
        self.clip_epsilon = 0.2
        self.lr = 0.0001  # Reduced from 0.0003 for stability
        self.hidden_size = 256
        self.max_grad_norm = 0.5
        self.value_loss_coef = 1.0  # Increased from 0.5 to stabilize critic
        self.entropy_coef = 0.01
        self.horizon = 2048  # Collect this many steps before update

        # Initialize network
        self.actor_critic = ActorCritic(self.num_states, self.num_actions, self.hidden_size).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr)

        # Learning rate scheduler - reduces LR over time
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.95
        )

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer()

        if path is not None:
            self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Loaded model from {path}")

    def get_action(self, state, deterministic=False):
        """Sample action from policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.actor_critic.get_action(state, deterministic)

        if deterministic:
            return action.cpu().numpy()[0], None, None
        else:
            return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0

        values = values + [next_value]

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).unsqueeze(1)
        returns = advantages + torch.FloatTensor(values[:-1]).unsqueeze(1)

        return advantages, returns

    def update(self):
        """Update PPO network"""
        states, actions, rewards, dones, old_log_probs, values = self.rollout_buffer.get()

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)

        # Compute advantages and returns
        with torch.no_grad():
            # Get value for last state
            last_state = states[-1].unsqueeze(0)
            _, _, next_value = self.actor_critic.forward(last_state)
            next_value = next_value.cpu().numpy()[0][0]

        advantages, returns = self.compute_gae(
            rewards.numpy(),
            values.numpy().squeeze().tolist(),
            dones.numpy(),
            next_value
        )

        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        dataset_size = states.size(0)
        indices = np.arange(dataset_size)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0

        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions
                log_probs, values, entropy = self.actor_critic.evaluate_actions(batch_states, batch_actions)

                # Policy loss with clipping
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()

        num_updates = self.n_epochs * (dataset_size // self.batch_size)
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy_loss = total_entropy_loss / num_updates

        return avg_policy_loss, avg_value_loss, avg_entropy_loss

    def train(self):
        episode_rewards = []
        running_reward = 0
        episode_count = 0
        total_steps = 0
        update_count = 0

        average = deque(maxlen=100)

        print("Starting PPO training on LunarLander-v3...")
        print(f"Hyperparameters: batch_size={self.batch_size}, lr={self.lr}, horizon={self.horizon}")

        state, info = self.env.reset()
        episode_reward = 0

        while True:
            # Collect rollout
            for _ in range(self.horizon):
                # Render environment if enabled
                if self.env.render_mode == 'human':
                    self.env.render()

                # Select action
                action, log_prob, value = self.get_action(state)

                # Execute action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Store transition
                self.rollout_buffer.add(state, action, reward, done, log_prob, value)

                episode_reward += reward
                state = next_state
                total_steps += 1

                if done:
                    episode_rewards.append(episode_reward)
                    average.append(episode_reward)
                    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
                    episode_count += 1

                    # Reset environment
                    state, info = self.env.reset()
                    episode_reward = 0

                    # Log progress
                    if episode_count % 10 == 0:
                        num_avg_episodes = min(len(average), 100)
                        avg_reward = np.mean(list(average)[-100:]) if len(average) > 0 else 0
                        print(f"Episode: {episode_count}, Running Reward: {running_reward:.2f}, "
                              f"Avg (last {num_avg_episodes} ep): {avg_reward:.2f}, Total Steps: {total_steps}")

                    # Save model every 100 episodes
                    if episode_count == 100:
                        save_path = f'PPO_LunarLander_ep{episode_count}.pth'
                        torch.save(self.actor_critic.state_dict(), save_path)
                        print(f"Model saved: {save_path}")
                    if episode_count % 1000 == 0:
                        save_path = f'PPO_LunarLander_ep{episode_count}.pth'
                        torch.save(self.actor_critic.state_dict(), save_path)
                        print(f"Model saved: {save_path}")

                    # Check if solved
                    if len(average) >= 100:
                        avg_reward = np.mean(list(average))
                        if avg_reward > 200:
                            print(f"Solved at episode {episode_count}! Average reward: {avg_reward:.2f}")
                            torch.save(self.actor_critic.state_dict(), 'PPO_LunarLander_final.pth')
                            return self.actor_critic

            # Update policy
            policy_loss, value_loss, entropy_loss = self.update()
            update_count += 1

            # Step learning rate scheduler
            self.scheduler.step()

            if update_count % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Update {update_count}: Policy Loss: {policy_loss:.4f}, "
                      f"Value Loss: {value_loss:.4f}, Entropy Loss: {entropy_loss:.4f}, LR: {current_lr:.6f}")

            # Clear rollout buffer
            self.rollout_buffer.clear()

    def test(self, path, num_episodes=10):
        """Test trained policy"""
        if path:
            self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Loaded model from {path}")

        total_rewards = []

        for episode in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0

            for step in range(self.max_steps_per_episode):
                if self.env.render_mode == 'human':
                    self.env.render()
                action, _, _ = self.get_action(state, deterministic=True)
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
        model_files = glob.glob('PPO_LunarLander_ep*.pth')

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
            self.actor_critic.load_state_dict(torch.load(model_file, map_location=self.device))
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

    ppo = PPO(args.path if args.mode == 'train' else None, render_mode=render_mode, device=args.device)

    if args.mode == 'train':
        ppo.train()
    elif args.mode == 'test':
        if args.path is None:
            print("Error: Please provide --path argument for testing a single model")
        else:
            ppo.test(args.path, num_episodes=args.episodes)
    elif args.mode == 'test_all':
        ppo.test_all_models(num_episodes=args.episodes)
    else:
        print(f"Unknown mode: {args.mode}. Use 'train', 'test', or 'test_all'")