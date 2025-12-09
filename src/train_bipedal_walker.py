import gymnasium as gym

import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env

    return thunk


def make_vec_env(env_id, n_envs=8, seed=0):
    return gym.vector.SyncVectorEnv([make_env(env_id, seed + i) for i in range(n_envs)])


# ---------------------------------------------------------------
# Parallel Generalized Advantage Estimation (GAE)
# ---------------------------------------------------------------
def compute_gae_parallel(rewards, values, dones, gamma, lam):
    """
    rewards: [T, N]
    values:  [T+1, N]
    dones:   [T, N]
    Returns:
        advantages: [T, N]
        returns:    [T, N]
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)

    gae = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns


# ---------------------------------------------------------------
#                PARALLEL A2C TRAINING LOOP
# ---------------------------------------------------------------
def train_parallel(
    env,
    model,
    steps=1_000_000,
    rollout_steps=256,
    gamma=0.99,
    lam=0.95,
    lr=3e-4,
    device="cpu",
):

    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_envs = env.num_envs
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32).to(device)

    global_step = 0
    episode_returns = np.zeros(n_envs)
    episode_lengths = np.zeros(n_envs)

    while global_step < steps:

        # ----- Buffers for vectorized rollout -----
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        done_buf = []
        val_buf = []

        # -----------------------------
        #   Rollout for rollout_steps
        # -----------------------------
        for _ in range(rollout_steps):

            with torch.no_grad():
                actions, logps, values = model(obs)
                values = values.squeeze(-1)

            # move to CPU numpy
            actions_np = actions.cpu().numpy()
            logps_np = logps.cpu().numpy()
            values_np = values.cpu().numpy()

            next_obs, rewards, dones, trunc, infos = env.step(actions_np)

            # Track episodic returns for logging
            episode_returns += rewards
            episode_lengths += 1

            for i in range(n_envs):
                if dones[i] or trunc[i]:
                    print(f"[Env {i}] Episode return: {episode_returns[i]:.1f}")
                    episode_returns[i] = 0
                    episode_lengths[i] = 0

            # Save buffers
            obs_buf.append(obs.cpu().numpy())
            act_buf.append(actions_np)
            logp_buf.append(logps_np)
            rew_buf.append(rewards)
            done_buf.append(dones)
            val_buf.append(values_np)

            # continue
            obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
            global_step += n_envs

        # convert to tensors / arrays
        obs_arr = np.array(obs_buf)  # [T, N, obs_dim]
        act_arr = np.array(act_buf)  # [T, N, act_dim]
        logp_arr = np.array(logp_buf)  # [T, N]
        rew_arr = np.array(rew_buf)  # [T, N]
        done_arr = np.array(done_buf)  # [T, N]
        val_arr = np.array(val_buf)  # [T, N]

        # bootstrap last value
        with torch.no_grad():
            last_val = model.get_value(obs).cpu().numpy()  # [N]
        val_arr = np.vstack([val_arr, last_val[None, :]])  # → [T+1, N]

        # ---------------------------------------------------
        #   Compute advantages + returns with GAE
        # ---------------------------------------------------
        adv_arr, ret_arr = compute_gae_parallel(rew_arr, val_arr, done_arr, gamma, lam)

        # flatten T,N → (T*N)
        T = rollout_steps
        obs_flat = obs_arr.reshape(T * n_envs, -1)
        act_flat = act_arr.reshape(T * n_envs, -1)
        logp_old_flat = logp_arr.reshape(T * n_envs)
        adv_flat = adv_arr.reshape(T * n_envs)
        ret_flat = ret_arr.reshape(T * n_envs)

        # normalize advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # convert to torch
        obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=device)
        act_t = torch.tensor(act_flat, dtype=torch.float32, device=device)
        logp_old_t = torch.tensor(logp_old_flat, dtype=torch.float32, device=device)
        adv_t = torch.tensor(adv_flat, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret_flat, dtype=torch.float32, device=device)

        # ---------------------------------------------------
        #                 A2C UPDATE (Single epoch)
        # ---------------------------------------------------
        # Compute loss on the whole batch (A2C is synchronous)
        logp, entropy, values = model.evaluate_actions(obs_t, act_t)
        values = values.squeeze(-1)

        actor_loss = -(logp * adv_t).mean()
        critic_loss = F.mse_loss(values, ret_t)
        entropy_loss = -0.0001 * entropy.mean()

        loss = actor_loss + critic_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[Step {global_step}] Updated policy | Loss = {loss.item():.4f}")

    return model
