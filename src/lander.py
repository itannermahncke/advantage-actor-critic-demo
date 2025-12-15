"""
Lunar-lander reinforcement learning environment:
Agent learns to land spacecraft

a2c: Agent uses Advantage Actor Critic algorithm

"""

import gymnasium as gym
from a2c import A2C
import torch.optim as optim
import numpy as np
from enum import Enum
import torch


class mode(Enum):
    """
    Mode indicates the style of rollout for this test.

    single_env: A3C with by-episode updates
    multi_env: A2C (vectorized env) with by-episode updates
    multi_env: A2C (vectorized env) with n-step updates
    """

    SINGLE_ENV = 0
    MULTI_ENV = 1
    N_STEP = 2


def log_episodic_info():
    # build the episode logging
    for step_idx in range(done_mat.shape[0]):  # Loop through steps
        for env_idx in range(envs.num_envs):  # Loop through envs
            # Accumulate reward for this step
            current_episode_rewards[env_idx] += rewards_mat[step_idx, env_idx].item()
            current_episode_lengths[env_idx] += 1

            # Check if episode ended
            if done_mat[step_idx, env_idx] > 0.5:
                # Episode finished!
                episode_rewards_buffer.append(current_episode_rewards[env_idx])
                episode_lengths_buffer.append(current_episode_lengths[env_idx])

                # Reset counters for this env
                current_episode_rewards[env_idx] = 0.0
                current_episode_lengths[env_idx] = 0


# essentials
LR = 0.0001
VALUE_LOSS_COEFF = 0.5
ENTROPY_COEF = 0.03
SEED = None
MODE = mode.N_STEP
MAX_EPISODES = 10000  # 10k
MAX_STEPS = 1000000000  # 1B

# vectorized environment for parallel learning
envs = gym.make_vec(
    "LunarLander-v3",
    num_envs=8,
    max_episode_steps=200,
    render_mode="rgb_array",
    vectorization_mode="sync",
    # autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
)

# logging
episode_rewards_buffer = []  # Store completed episode rewards
episode_lengths_buffer = []
current_episode_rewards = np.zeros(envs.num_envs)  # Track ongoing episodes
current_episode_lengths = np.zeros(envs.num_envs, dtype=int)

# actor-critic agent
agent = A2C(envs, random_seed=SEED, gamma=0.99, device="cpu")

# optimizers
actor_optim = optim.Adam(agent.actor.parameters(), lr=LR)
critic_optim = optim.Adam(agent.critic.parameters(), lr=LR)

# reward tracking
reward_history = []
max_reward = float("-inf")

match MODE:
    # episodic updates -- slower if an env is taking a long time
    case mode.MULTI_ENV:
        # 1. TRAINING LOOP
        for episode in range(MAX_EPISODES):
            # 1.1 Clear Gradients
            actor_optim.zero_grad()
            critic_optim.zero_grad()

            # 1.2 Unit Training
            returns, values, log_probs, episode_rewards = agent.train_by_episode()
            avg_reward = episode_rewards.mean().item()
            reward_history.append(avg_reward)

            # 1.3 Print Info
            if episode % 100 == 0:
                recent_avg = sum(reward_history[-100:]) / len(reward_history[-100:])
                print(
                    f"Episode {episode}: mean reward = {avg_reward:.2f}, "
                    f"100-episode avg = {recent_avg:.2f}"
                )

            # Compute returns + advantages via GAE
            # returns, advantages = agent.compute_gae_returns(
            #     rewards_mat, values_mat, done_mat, next_value
            # )

            # 1.4 Compute Loss of Actor and Critic
            actor_loss, critic_loss = agent.compute_loss(
                log_probs,
                returns,
                values,
            )

            # 1.5 Backpropagation and Optimization
            actor_loss.backward()
            critic_loss.backward()
            actor_optim.step()
            critic_optim.step()

    # batch updates -- episodes get interrupted
    case mode.N_STEP:
        # 1. TRAINING LOOP
        for s in range(MAX_STEPS):
            # 1.1 Clear Old Gradients
            actor_optim.zero_grad()
            critic_optim.zero_grad()

            # 1.2 Unit Training
            (
                obs_mat,
                actions_mat,
                rewards_mat,
                values_mat,
                logprob_mat,
                entropy_mat,
                done_mat,
                next_value,
            ) = agent.train_by_n_step_rollout()
            log_episodic_info()

            # 1.3 Print Info
            if s % 1000 == 0:
                # Episode statistics (if any episodes completed)
                if len(episode_rewards_buffer) > 100:
                    recent_episodes = episode_rewards_buffer[-100:]  # Last 100 episodes
                    mean_ep_reward = np.mean(recent_episodes)
                    max_ep_reward = np.max(episode_rewards_buffer)
                    num_episodes = len(episode_rewards_buffer)
                    recent_lengths = episode_lengths_buffer[-100:]
                    mean_length = np.mean(recent_lengths)
                    mean_entropy = entropy_mat.mean().item()

                    print(
                        f"Step {s}: "
                        f"Episodes: {num_episodes}, "
                        f"Mean reward (last 100): {mean_ep_reward:.2f}, "
                        f"Mean length: {mean_length:.1f}, "
                        f"Mean entropy: {mean_entropy:.1f}, "
                        f"Max ever: {max_ep_reward:.2f}"
                    )

                    # agent success criteria
                    if np.mean(mean_ep_reward) >= 200:
                        print(
                            f"SOLVED! Mean over last 100 episodes: {np.mean(mean_ep_reward):.2f}"
                        )
                        torch.save(
                            {
                                "step": s,
                                "actor_state_dict": agent.actor.state_dict(),
                                "critic_state_dict": agent.critic.state_dict(),
                                "actor_optimizer": actor_optim.state_dict(),
                                "critic_optimizer": critic_optim.state_dict(),
                                "episode_rewards": episode_rewards_buffer,
                            },
                            f"trained_a2c_agent.pt",
                        )
                        print(f"Checkpoint saved at step {s}")
                else:
                    print(f"Step {s}: " f"No episodes completed yet")

            # 1.4 Compute Loss for Actor and Critic
            returns, advantages = agent.compute_gae_returns(
                rewards_mat, values_mat, done_mat, next_value
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            actor_loss, critic_loss = agent.compute_loss(
                logprob_mat,
                returns,
                values_mat,
                advantages,
            )

            # Backpropagation and Optimization
            entropy_loss = entropy_mat.mean()
            actor_total_loss = actor_loss - ENTROPY_COEF * entropy_loss
            critic_total_loss = VALUE_LOSS_COEFF * critic_loss
            actor_total_loss.backward()
            critic_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
            actor_optim.step()
            critic_optim.step()
