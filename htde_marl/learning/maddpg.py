import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from htde_marl.learning.replay_buffer import ReplayBuffer


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    """Centralised critic: takes all agents' obs + actions concatenated."""
    def __init__(self, total_obs_dim: int, total_act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs_all, acts_all):
        x = torch.cat([obs_all, acts_all], dim=-1)
        return self.net(x)


class MADDPGAgent:
    def __init__(self, agent_id: int, obs_dim: int, act_dim: int,
                 total_obs_dim: int, total_act_dim: int,
                 lr_actor: float = 1e-4, lr_critic: float = 3e-4,
                 gamma: float = 0.99, tau: float = 0.005):
        self.agent_id = agent_id
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau

        self.actor        = Actor(obs_dim, act_dim)
        self.actor_target = Actor(obs_dim, act_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic        = Critic(total_obs_dim, total_act_dim)
        self.critic_target = Critic(total_obs_dim, total_act_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def act(self, obs: np.ndarray, mask: np.ndarray = None, noise: float = 0.1) -> int:
        with torch.no_grad():
            probs = self.actor(torch.FloatTensor(obs)).numpy()
        if mask is not None:
            probs = probs * mask
            if probs.sum() == 0:
                probs = mask.astype(float)
            probs /= probs.sum()
        # add exploration noise via Gumbel
        if noise > 0:
            gumbel = -np.log(-np.log(np.random.rand(*probs.shape) + 1e-9) + 1e-9)
            probs = np.exp((np.log(probs + 1e-9) + gumbel * noise))
            probs /= probs.sum()
        return int(np.argmax(probs))

    def update(self, batch: list, all_agents: list, obs_dim: int):
        obs_all   = torch.FloatTensor(np.array([t["obs_all"]      for t in batch]))
        acts_all  = torch.FloatTensor(np.array([t["acts_all"]     for t in batch]))
        obs_next  = torch.FloatTensor(np.array([t["obs_next_all"] for t in batch]))
        reward    = torch.FloatTensor(np.array([t["rewards"][self.agent_id] for t in batch])).unsqueeze(1)
        done      = torch.FloatTensor(np.array([t["done"]         for t in batch])).unsqueeze(1)

        def slice_obs(o, idx): return o[:, idx*obs_dim:(idx+1)*obs_dim]

        # --- critic update ---
        with torch.no_grad():
            next_acts = torch.cat([
                a.actor_target(slice_obs(obs_next, i))
                for i, a in enumerate(all_agents)
            ], dim=-1)
            target_q = reward + self.gamma * (1 - done) * self.critic_target(obs_next, next_acts)

        current_q = self.critic(obs_all, acts_all)
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_opt.zero_grad(); critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # --- actor update ---
        new_act_i    = self.actor(slice_obs(obs_all, self.agent_id))
        new_acts_all = acts_all.clone()
        new_acts_all[:, self.agent_id * self.act_dim:(self.agent_id + 1) * self.act_dim] = new_act_i
        actor_loss = -self.critic(obs_all, new_acts_all).mean()
        self.actor_opt.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, src: nn.Module, tgt: nn.Module):
        for sp, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
