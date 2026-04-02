import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from htde_marl.env.htde_env import HTDEEnv
from htde_marl.agents.planner import PlannerAgent
from htde_marl.learning.maddpg import MADDPGAgent
from htde_marl.learning.replay_buffer import ReplayBuffer
from htde_marl.experiments.train import executor_obs_vec


def train_maddpg(
    num_tasks: int = 8,
    num_executors: int = 2,
    complexity: str = "medium",
    episodes: int = 500,
    batch_size: int = 128,
    warmup_steps: int = 500,
    update_every: int = 4,
    seed: int = 0,
):
    np.random.seed(seed)
    env = HTDEEnv(num_tasks=num_tasks, num_executors=num_executors,
                  complexity=complexity, max_steps=50)
    planner = PlannerAgent(policy="heuristic")

    obs_dim     = num_tasks * 4          # status + mask + features(2)
    act_dim     = num_tasks + 1          # tasks + idle
    total_obs   = obs_dim * num_executors
    total_act   = act_dim * num_executors

    agents = [
        MADDPGAgent(i, obs_dim, act_dim, total_obs, total_act)
        for i in range(num_executors)
    ]
    buffer = ReplayBuffer(capacity=50000)
    ep_rewards = []
    step_count = 0

    for ep in range(episodes):
        obs = env.reset()
        ep_total = 0.0

        for _ in range(50):
            scores = planner.act(obs["planner"])
            vecs  = [executor_obs_vec(obs[f"executor_{i}"]) for i in range(num_executors)]
            masks = [np.append(obs[f"executor_{i}"]["available_mask"], 1) for i in range(num_executors)]

            noise = max(0.05, 1.0 - ep / (episodes * 0.6))
            raw_acts = [agents[i].act(vecs[i], masks[i], noise) for i in range(num_executors)]

            actions = {"planner": scores}
            for i in range(num_executors):
                actions[f"executor_{i}"] = raw_acts[i] if raw_acts[i] < num_tasks else -1

            next_obs, rewards, done, _ = env.step(actions)
            ep_total += sum(rewards.values())

            next_vecs = [executor_obs_vec(next_obs[f"executor_{i}"]) for i in range(num_executors)]

            # one-hot encode actions for critic input
            def one_hot(a, n):
                v = np.zeros(n, dtype=np.float32); v[a] = 1.0; return v

            buffer.push({
                "obs_all":      np.concatenate(vecs),
                "acts_all":     np.concatenate([one_hot(a, act_dim) for a in raw_acts]),
                "obs_next_all": np.concatenate(next_vecs),
                "rewards":      {i: rewards[f"executor_{i}"] for i in range(num_executors)},
                "done":         float(done),
            })
            step_count += 1
            obs = next_obs

            if len(buffer) >= batch_size and step_count >= warmup_steps and step_count % update_every == 0:
                batch = buffer.sample(batch_size)
                for agent in agents:
                    agent.update(batch, agents, obs_dim)

            if done:
                obs = env.reset()
                break

        ep_rewards.append(ep_total)
        if (ep + 1) % 50 == 0:
            avg = np.mean(ep_rewards[-50:])
            print(f"Episode {ep+1:4d} | avg_reward (last 50): {avg:.1f}")

    return agents, ep_rewards


if __name__ == "__main__":
    train_maddpg(episodes=300)
