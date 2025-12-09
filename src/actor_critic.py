import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=256):
        super().__init__()

        # ---- Shared feature encoder ----
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # ---- Actor head (outputs mean of Gaussian) ----
        self.actor_mean = nn.Linear(hidden_size, action_dim)

        # Learnable log_std for each action dimension
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # ---- Critic head ----
        self.critic = nn.Linear(hidden_size, 1)

        # Optional: orthogonal initialization
        self._init_weights()

    def _init_weights(self):
        # Orthogonal initialization improves stability
        for layer in self.feature_net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.constant_(layer.bias, 0)

        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0)

    # -------------------------------------------------
    #          Core forward functions
    # -------------------------------------------------

    def forward(self, obs):
        """Return action, log_prob, and value when acting in the environment."""
        features = self.feature_net(obs)

        # Critic
        value = self.critic(features)

        # Actor
        mean = self.actor_mean(features)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)

        # Sample action
        action = dist.rsample()  # rsample → reparameterized, good for backprop
        log_prob = dist.log_prob(action).sum(axis=-1)

        # Squash actions with tanh for [-1, 1] range (BipedalWalker requirement)
        action_tanh = torch.tanh(action)

        # Return raw + squashed action?
        # Most algorithms use squashed actions, so:
        return action_tanh, log_prob, value

    def get_value(self, obs):
        """Compute only the critic value."""
        features = self.feature_net(obs)
        return self.critic(features)

    def evaluate_actions(self, obs, actions):
        """
        Used during training to compute:
        - log_probs of the actions that were actually taken
        - entropy of the policy
        - values at those states
        """
        features = self.feature_net(obs)

        mean = self.actor_mean(features)
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)

        # Because actions were squashed, we unsquash them
        # arctanh(x) = 0.5 * ln((1 + x) / (1 - x))
        # but must clip to avoid nan
        eps = 1e-6
        clipped = torch.clamp(actions, -1 + eps, 1 - eps)
        unsquashed = 0.5 * torch.log((1 + clipped) / (1 - clipped))

        log_prob = dist.log_prob(unsquashed).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)

        value = self.critic(features)

        return log_prob, entropy, value
