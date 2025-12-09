from actor_critic import ActorCritic
from train_bipedal_walker import make_vec_env, train_parallel

env = make_vec_env("BipedalWalker-v3", n_envs=8, seed=0)

model = ActorCritic(
    obs_dim=env.single_observation_space.shape[0],
    action_dim=env.single_action_space.shape[0],
).to("cpu")

train_parallel(env, model, steps=500_000, rollout_steps=256, device="cpu")
