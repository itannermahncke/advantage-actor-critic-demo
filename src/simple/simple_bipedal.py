"""
Bipedal walker reinforcement learning environment:
Agent learns to walk

a2c: Agent uses Advantage Actor Critic algorithm

"""

import logging
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
from actor_critic_simple import A2C
import torch.optim as optim
import math

LR = 0.001  # Learning rate
MAX_EPISODES = 10000  # Max number of episodes

env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
# Record videos periodically (every 250 episodes)
env = RecordVideo(
    env,
    video_folder="videos",
    name_prefix="training",
    episode_trigger=lambda x: x % 100 == 0,  # Only record every 250th episode
)

# Track statistics for every episode (lightweight)
env = RecordEpisodeStatistics(env)

# Init actor-critic agent
agent = A2C(env, gamma=0.999)


# Init optimizers
actor_optim = optim.Adam(agent.actor.parameters(), lr=LR)
critic_optim = optim.Adam(agent.critic.parameters(), lr=LR)

#
# Train
#

r = []  # Array containing total rewards
avg_r = 0  # Value storing average reward over last 100 episodes
max_r = -math.inf

for i in range(MAX_EPISODES):
    obs, info = env.reset()
    critic_optim.zero_grad()
    actor_optim.zero_grad()

    info, rewards, critic_vals, action_lp_vals, total_reward = agent.train_env_episode(
        render=False
    )

    r.append(total_reward)

    # Check if we won the game
    if total_reward >= 300:
        print("solved")
        break

    if "episode" in info:
        episode_data = info["episode"]
        logging.info(
            f"Episode {i}: "
            f"reward={episode_data['r']:.1f}, "
            f"length={episode_data['l']}, "
            f"time={episode_data['t']:.2f}s"
        )

    # Check average reward every 100 episodes
    if len(r) >= 100:
        episode_count = i - (i % 100)
        prev_episodes = r[len(r) - 100 :]
        avg_r = sum(prev_episodes) / len(prev_episodes)
        if len(r) % 100 == 0:
            print(
                f"Average reward during episodes {episode_count}-{episode_count + 100} is {avg_r}"
            )
            RecordVideo(env, "./videos")

    l_actor, l_critic = agent.compute_loss(
        action_p_vals=action_lp_vals, G=rewards, V=critic_vals
    )

    l_actor.backward()
    l_critic.backward()

    actor_optim.step()
    critic_optim.step()

#
# Test
#
for _ in range(10):
    agent.test_env_episode(render=True)
