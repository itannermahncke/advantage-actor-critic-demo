import torch
import torch.nn as nn
from torch.distributions import Categorical

import numpy as np
import logging
import gymnasium as gym


class A2C(nn.Module):
    """
    Docstring for A2C class.
    """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        hidden_size=128,
        gamma=0.99,
        random_seed=None,
        device="cpu",
    ):
        """
        Initialize an actor-critic instance.
        env: reference vectorized environments
        """
        super().__init__()

        # variables
        self.env = env
        obs, _ = env.reset(seed=random_seed)
        self.num_envs = env.num_envs
        self.gamma = gamma  # discount factor for rewards
        self.hidden_size = hidden_size  # num. neurons in NN
        self.device = device

        # seed for reproducability
        if random_seed is not None:
            torch.manual_seed(random_seed)

        # dimensions of inputs and outputs
        self.in_size = (
            np.array(obs[0], dtype=np.float32).flatten().shape[0]
        )  # dimensions of an observation
        self.out_size = env.single_action_space.n  # number of available actions

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.out_size),
        ).to(device)

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),  # value estimate
        ).to(device)

    def train_envs_episode(self, ep_num=None):
        """
        Collects synchronous rollouts until all envs finish their episode.

        Returns:
            rewards:   (T, num_envs) discounted returns
            values:    (T, num_envs)
            log_probs: (T, num_envs)
            total_reward: array of episode rewards of shape (num_envs,)
        """
        # reset at the start of an episode
        obs, info = self.env.reset()
        done_flags = np.zeros(self.num_envs, dtype=bool)

        # accumulate episode reward
        rewards_per_env = np.zeros(self.num_envs)

        # make lists
        rewards_list = []
        values_list = []
        logprob_list = []

        # wait for all envs to finish their episode
        while not np.all(done_flags):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

            # Actor makes a choice
            logits = self.actor(obs_t)  # (num_envs, num_actions)
            dist = Categorical(logits=logits)
            actions = dist.sample()  # (num_envs,)
            log_probs = dist.log_prob(actions)  # (num_envs,)

            # Critic evaluates the current state
            values = self.critic(obs_t).squeeze(-1)  # (num_envs,)

            # Step vectorized env and see if env is done
            obs_next, reward, terminated, truncated, info = self.env.step(
                actions.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)

            # Accumulate episode reward
            rewards_per_env += reward * (~done_flags)

            # Store rollout step
            rewards_list.append(
                torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            )
            values_list.append(values)
            logprob_list.append(log_probs)

            # End an env only if done
            done_flags = np.logical_or(done_flags, done)

            obs = obs_next

        # now all envs are done -- learn for next episode

        # Compute discounted returns *per environment*
        # rewards_list is list of [num_envs] tensors
        # Convert to (T, num_envs) tensor
        rewards_mat = torch.stack(rewards_list)  # (T, num_envs)
        values_mat = torch.stack(values_list)  # (T, num_envs)
        logprob_mat = torch.stack(logprob_list)  # (T, num_envs)

        T = rewards_mat.shape[0]

        # weight later rewards with discount factor via backwards recursion
        returns = torch.zeros_like(rewards_mat)
        G = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(T)):
            G = rewards_mat[t] + self.gamma * G
            returns[t] = G

        # Normalize returns so there isn't bias based on side of the envs vector
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Logging (optional)
        if ep_num is not None and "episode" in info:
            episode_data = info["episode"]
            logging.info(
                f"Episode {ep_num}: "
                f"reward={episode_data['r']:.1f}, "
                f"length={episode_data['l']}, "
                f"time={episode_data['t']:.2f}s"
            )

        return returns, values_mat, logprob_mat, rewards_per_env.copy()

    def test_env_episode(self, render=True):
        """
        Runs ONE of the vectorized envs (env index 0). For testing the agent.

        Returns total reward for env 0 only.
        """

        obs, info = self.env.reset()

        total_reward = 0.0
        done = False

        # Only track env 0
        while not done:
            if render:
                self.env.render()

            obs0 = obs[0]  # get first env

            obs_t = torch.as_tensor(obs0, dtype=torch.float32, device=self.device)
            logits = self.actor(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()

            obs_next, reward, terminated, truncated, info = self.env.step(
                np.array([action.item()] + [0] * (self.num_envs - 1))
            )

            done = terminated[0] or truncated[0]
            total_reward += reward[0]

            obs = obs_next

        return total_reward

    @staticmethod
    def compute_loss(log_probs, returns, values, critic_loss=nn.SmoothL1Loss()):
        """
        Compute updates for the actor and the critic.

        The actor updates its theta parameter based on rewards and the critic's feedback, which is tuned to approximate the policy function.

        The critic's loss comes from the difference between the values it predicted for the actor's action, and the actual returns producted by that action.

        log_probs: (T, num_envs)
        returns:   (T, num_envs)
        values:    (T, num_envs)
        """
        advantage = returns - values.detach()

        # mean makes the actor loss invariant to the number of environments training at once
        actor_loss = -(log_probs * advantage).mean()
        critic_loss_val = critic_loss(values, returns)

        return actor_loss, critic_loss_val
