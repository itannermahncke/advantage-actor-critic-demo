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
        env: gym.vector.VectorEnv | gym.Env,
        batch_steps=5,
        hidden_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        random_seed=None,
        device="cpu",
    ):
        """
        Initialize an actor-critic instance.
        env: reference vectorized environments
        """
        # super
        super().__init__()

        # variables
        self.env = env
        obs, _ = env.reset(seed=random_seed)
        self.current_obs = None

        if isinstance(env, gym.Env):
            self.num_envs = 1
        else:
            self.num_envs = env.num_envs

        self.gamma = gamma  # discount factor for rewards
        self.gae_lambda = (
            gae_lambda  # how much do we care about earlier advantage estimates
        )
        self.batch_steps = (
            batch_steps  # num. of episodes to run before calculating returns
        )
        self.hidden_size = hidden_size  # num. neurons in NN
        self.device = device  # cpu or gpu

        # seed for reproducability
        if random_seed is not None:
            torch.manual_seed(random_seed)

        # dimensions of inputs and outputs
        self.in_size = (
            np.array(obs[0], dtype=np.float32).flatten().shape[0]
        )  # dimensions of an observation
        self.out_size = env.single_action_space.n  # number of available actions

        # actor network
        self.actor = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.out_size),
        ).to(device)

        # critic network
        self.critic = nn.Sequential(
            nn.Linear(self.in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),  # value estimate
        ).to(device)

    def train_by_n_step_rollout(self):
        """
        Roll out all vectorized environments for a batch of steps.

        Returns:
            obs:        (n_steps, num_envs, obs_dim)
            actions:    (n_steps, num_envs)
            rewards:    (n_steps, num_envs)
            values:     (n_steps, num_envs)
            log_probs:  (n_steps, num_envs)
            entropies:
            dones:      (n_steps, num_envs)
            next_value: (num_envs,)   critic bootstrap
        """

        if self.num_envs == 1:
            print("Calling multi-env episode on a single env!")
            return None

        # ---- rollout storage ----
        obs_list = []
        actions_list = []
        rewards_list = []
        values_list = []
        logprob_list = []
        entropy_list = []
        done_list = []

        # ---- pick up where we left off ----
        if self.current_obs is None:
            obs, _ = self.env.reset()
        else:
            obs = self.current_obs

        # main rollout loop
        for _ in range(self.batch_steps):

            # convert obs to tensor
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

            # store observation (minimal diff: just append original tensor)
            obs_list.append(obs_t)

            # ---- Actor ----
            logits = self.actor(obs_t)
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()

            # store actions
            actions_list.append(actions)
            entropy_list.append(entropies)

            # ---- Critic ----
            values = self.critic(obs_t).squeeze(-1)

            # ---- Step all environments ----
            next_obs, rewards, terminated, truncated, info = self.env.step(
                actions.cpu().numpy()
            )
            dones = np.logical_or(terminated, truncated)

            # ---- Store step data ----
            rewards_list.append(
                torch.tensor(rewards, dtype=torch.float32, device=self.device)
            )
            values_list.append(values)
            logprob_list.append(log_probs)
            done_list.append(
                torch.tensor(dones, dtype=torch.float32, device=self.device)
            )

            # advance obs
            obs = next_obs

        # ---- store latest obs for next rollout ----
        # minimal change: ensure tensor conversion
        self.current_obs = obs

        # ---- Bootstrap next-value (critic on final obs) ----
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            next_value = self.critic(obs_t).squeeze(-1)  # (num_envs,)

        # ---- Stack rollout arrays ----
        obs_mat = torch.stack(obs_list)  # (n_steps, num_envs, obs_dim)
        actions_mat = torch.stack(actions_list)  # (n_steps, num_envs)
        rewards_mat = torch.stack(rewards_list)  # (n_steps, num_envs)
        values_mat = torch.stack(values_list)  # (n_steps, num_envs)
        logprob_mat = torch.stack(logprob_list)  # (n_steps, num_envs)
        entropy_mat = torch.stack(entropy_list)
        done_mat = torch.stack(done_list)  # (n_steps, num_envs)

        return (
            obs_mat,
            actions_mat,
            rewards_mat,
            values_mat,
            logprob_mat,
            entropy_mat,
            done_mat,
            next_value,
        )

    def train_by_episode(self, ep_num=None):
        """
        Collects synchronous results until all envs finish their episode.

        Returns:
            rewards:   (T, num_envs) discounted returns
            values:    (T, num_envs)
            log_probs: (T, num_envs)
            total_reward: array of episode rewards of shape (num_envs,)
        """
        # make sure we have a set of envs
        if self.num_envs == 1:
            print("Calling multi-env episode on a single env!")
            return None

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

    def train_single_env_episode(self, ep_num, render=False):
        """
        Docstring for train_single_env_episode. Train a single environment in a single episode.
        """
        # make sure we have the right kind of env
        if self.num_envs != 1:
            print("Calling single-env episode on a parallel set of envs!")
            return None

        rewards = []
        critic_vals = []
        log_probs = []
        obs, info = self.env.reset()
        done = False

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            # actor selects an action
            logits = self.actor(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # critic evaluates the state
            value = self.critic(obs_t).squeeze(-1)

            # note action probability and critic value
            log_probs.append(log_prob)
            critic_vals.append(value)

            # take a step in the environment
            obs, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            # note the reward earned
            rewards.append(
                torch.tensor(reward, dtype=torch.float32, device=self.device)
            )
            total_reward = sum([r.item() for r in rewards])

            # apply discount factor in reverse order
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + self.gamma * G
                returns.append(G)
            returns.reverse()
            returns = torch.stack(returns)

            # Standardize returns (helps training stability)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            critic_vals = torch.stack(critic_vals)
            log_probs = torch.stack(log_probs)
            return returns, critic_vals, log_probs, total_reward

    def test_env_episode(self, render=True):
        """
        Runs an episode of one environment. For demonstrating the agent.

        Returns total reward for env 0.
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

            obs_next, reward, terminated, truncated, _ = self.env.step(
                np.array([action.item()] + [0] * (self.num_envs - 1))
            )

            done = terminated[0] or truncated[0]
            total_reward += reward[0]

            obs = obs_next

        return total_reward

    def compute_gae_returns(self, rewards, values, dones, next_value):
        """
        Compute returns and advantages using Generalized Advantage Estimation.

        Args:
            rewards: (n_steps, num_envs)
            values: (n_steps, num_envs)
            dones: (n_steps, num_envs)
            next_value: (num_envs,)

        Returns:
            returns: (n_steps, num_envs)
            advantages: (n_steps, num_envs)
        """
        n_steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)

        # Start with the bootstrap value
        last_gae = torch.zeros_like(next_value)

        # Compute GAE backwards through time
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            # Mask out advantages for done states
            next_non_terminal = 1.0 - dones[t]

            # TD error: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]

            # GAE: A_t = delta_t + (gamma * lambda) * A_{t+1}
            advantages[t] = last_gae = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            )

        # Returns are advantages + values
        returns = advantages + values

        return returns, advantages

    @staticmethod
    def compute_loss(
        log_probs,
        returns,
        values,
        advantages=None,
        critic_loss_fn=nn.SmoothL1Loss(),
    ):
        """
        Compute updates for the actor and the critic.

        The actor updates its theta parameter based on rewards and the critic's feedback, which is tuned to approximate the policy function.

        The critic's loss comes from the difference between the values it predicted for the actor's action, and the actual returns producted by that action.

        log_probs:  (T, num_envs)
        returns:    (T, num_envs)
        values:     (T, num_envs)
        advantages: (T, num_envs)
        """
        if advantages is None:
            advantages = returns - values.detach()

        # Actor loss: policy gradient with advantage
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Critic loss: MSE between predicted values and returns
        critic_loss_val = critic_loss_fn(values, returns)

        return actor_loss, critic_loss_val
