import torch
import torch.nn as nn
from torch.distributions import Normal

import gymnasium as gym


class A2C(nn.Module):
    """
    An implementation of the Advantage Actor Critic (A2C) algorithm. The actor learns a policy function and executes actions, while the critic learns a value function and evaluates the actor's choices.

    Attributes:
        env: BipedalWalker environment
        self.gamma (float)
    """

    def __init__(self, env, hidden_size=128, gamma=0.99):
        """
        Initialize an instance of the A2C class.
        """

        super().__init__()

        self.env = env
        self.gamma = gamma
        self.hidden_size = hidden_size

        # Observation space may be Box([...])
        obs_sample = env.observation_space.sample()
        self.in_size = len(obs_sample.flatten())

        # Action space is continuous Box(shape=[n])
        assert hasattr(
            env.action_space, "shape"
        ), "Environment action space must be continuous (Box)"
        self.action_dim = env.action_space.shape[0]

        # ----------------------
        # Actor network
        # ----------------------
        self.actor = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_dim),
        ).double()

        # Learnable log std dev for each action dimension
        self.log_std = nn.Parameter(torch.zeros(self.action_dim, dtype=torch.double))

        # ----------------------
        # Critic network
        # ----------------------
        self.critic = nn.Sequential(
            nn.Linear(self.in_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        ).double()

    # ------------------------------------------------
    #   Run a single training episode
    # ------------------------------------------------
    def train_env_episode(self, render=False):
        rewards = []
        critic_vals = []
        action_lp_vals = []

        obs, info = self.env.reset()
        done = False

        while not done:
            if render:
                self.env.render()

            obs_t = torch.from_numpy(obs).double()

            # Actor forward pass → mean
            mean = self.actor(obs_t)
            std = torch.exp(self.log_std)

            # Gaussian distribution
            dist = Normal(mean, std)

            # Sample action
            raw_action = dist.rsample()  # rsample() supports gradients
            action = torch.tanh(raw_action)  # map to [-1,1]

            # log_prob of raw action BEFORE tanh (simple method)
            log_prob = dist.log_prob(raw_action).sum()

            # Critic value
            value = self.critic(obs_t).squeeze()

            # Store values
            action_lp_vals.append(log_prob)
            critic_vals.append(value)

            # Step environment
            action_env = action.detach().numpy()
            obs, reward, terminated, truncated, info = self.env.step(action_env)
            done = terminated or truncated

            rewards.append(torch.tensor(reward, dtype=torch.double))

        # Compute episodic returns (Monte Carlo)
        for t_i in range(len(rewards)):
            G = 0
            for t in range(t_i, len(rewards)):
                G += rewards[t] * (self.gamma ** (t - t_i))
            rewards[t_i] = G

        # Convert to tensors
        def stack(x):
            return torch.stack(tuple(x), 0)

        G = stack(rewards)
        V = stack(critic_vals)
        logps = stack(action_lp_vals)

        # Standardize returns
        G = (G - G.mean()) / (G.std() + 1e-8)

        total_reward = G[0].item()  # useful for logging
        return info, G, V, logps, total_reward

    # ------------------------------------------------
    #   Test episode (no gradients)
    # ------------------------------------------------
    def test_env_episode(self, render=True):
        obs, _ = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            if render:
                self.env.render()

            obs_t = torch.from_numpy(obs).double()

            mean = self.actor(obs_t)
            std = torch.exp(self.log_std)
            dist = Normal(mean, std)

            raw_action = dist.mean
            action = torch.tanh(raw_action)

            obs, reward, terminated, truncated, info = self.env.step(
                action.detach().numpy()
            )
            done = terminated or truncated
            total_reward += reward

        return total_reward

    # ------------------------------------------------
    #   Same loss function structure as your original
    # ------------------------------------------------
    @staticmethod
    def compute_loss(action_p_vals, G, V, critic_loss=nn.SmoothL1Loss()):
        assert len(action_p_vals) == len(G) == len(V)

        advantage = G - V.detach()
        actor_loss = -(action_p_vals * advantage).sum()
        critic_loss = critic_loss(G, V)

        return actor_loss, critic_loss
