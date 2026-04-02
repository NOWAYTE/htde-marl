import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.net(x)
        return self.actor(h), self.critic(h)

    def act(self, x, mask=None):
        logits, value = self(x)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(-1)


class PPOAgent:
    def __init__(self, obs_dim: int, action_dim: int,
                 lr: float = 3e-4, gamma: float = 0.99,
                 clip_eps: float = 0.2, epochs: int = 4,
                 hidden: int = 64):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.policy = PolicyNet(obs_dim, action_dim, hidden)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def act(self, obs_vec: np.ndarray, mask: np.ndarray = None):
        x = torch.FloatTensor(obs_vec)
        m = torch.FloatTensor(mask) if mask is not None else None
        with torch.no_grad():
            action, log_prob, value = self.policy.act(x, m)
        return action, log_prob.item(), value.item()

    def update(self, rollout: dict):
        obs    = torch.FloatTensor(np.array(rollout["obs"]))
        acts   = torch.LongTensor(rollout["actions"])
        old_lp = torch.FloatTensor(rollout["log_probs"])
        rets   = torch.FloatTensor(self._returns(rollout["rewards"], rollout["dones"]))
        masks  = torch.FloatTensor(np.array(rollout["masks"])) if rollout.get("masks") else None

        for _ in range(self.epochs):
            logits, values = self.policy(obs)
            if masks is not None:
                logits = logits.masked_fill(masks == 0, -1e9)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(acts)
            entropy = dist.entropy().mean()

            advantages = rets - values.squeeze(-1).detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = (log_probs - old_lp).exp()
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages

            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = (rets - values.squeeze(-1)).pow(2).mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def _returns(self, rewards, dones):
        returns, R = [], 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
        return returns
