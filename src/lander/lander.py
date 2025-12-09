"""
Lunar-lander reinforcement learning environment:
Agent learns to land spacecraft

a2c: Agent uses Advantage Actor Critic algorithm

"""

import logging
import gymnasium as gym
from a2c import A2C
import torch.optim as optim
import math

# essentials
LR = 0.001
SEED = None
MAX_EPISODES = 10000

# vectorized environment for parallel learning
envs = gym.make_vec(
    "LunarLander-v3",
    num_envs=8,
    render_mode="rgb_array",
    vectorization_mode="sync",
)
# logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# actor-critic agent
agent = A2C(envs, random_seed=SEED, gamma=0.999, device="cpu")

# optimizers
actor_optim = optim.Adam(agent.actor.parameters(), lr=LR)
critic_optim = optim.Adam(agent.critic.parameters(), lr=LR)

# reward tracking
reward_history = []
max_reward = float("-inf")

# start training!
for episode in range(MAX_EPISODES):
    actor_optim.zero_grad()
    critic_optim.zero_grad()

    # Vectorized rollout (collects T steps from all envs)
    returns, values, log_probs, episode_rewards = agent.train_envs_episode()
    avg_reward = episode_rewards.mean().item()
    reward_history.append(avg_reward)

    if episode % 100 == 0:
        recent_avg = sum(reward_history[-100:]) / len(reward_history[-100:])
        print(
            f"Episode {episode}: mean reward = {avg_reward:.2f}, "
            f"100-episode avg = {recent_avg:.2f}"
        )

    # train the actor and the critic
    actor_loss, critic_loss = agent.compute_loss(
        log_probs,
        returns,
        values,
    )

    # backpropagation: compute weight adjustments of the neural networks
    actor_loss.backward()
    critic_loss.backward()

    # optimization: apply weight adjustments to the neural networks
    actor_optim.step()
    critic_optim.step()
