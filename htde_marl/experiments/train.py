import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from htde_marl.env.htde_env import HTDEEnv
from htde_marl.agents.planner import PlannerAgent
from htde_marl.learning.ppo_multiagent import PPOAgent


# --- obs flattening helpers ---

def planner_obs_vec(obs: dict, num_tasks: int) -> np.ndarray:
    adj  = obs["adj_matrix"].flatten()
    stat = obs["task_status"].astype(np.float32)
    feat = obs["task_features"].flatten()
    t    = np.array([obs["time_step"] / 50.0], dtype=np.float32)
    return np.concatenate([adj, stat, feat, t])

def executor_obs_vec(obs: dict) -> np.ndarray:
    stat = obs["task_status"].astype(np.float32)
    mask = obs["available_mask"].astype(np.float32)
    feat = obs["task_features"].flatten()
    return np.concatenate([stat, mask, feat])


def train_ppo(
    num_tasks: int = 8,
    num_executors: int = 2,
    complexity: str = "medium",
    episodes: int = 500,
    rollout_len: int = 20,
    seed: int = 0,
):
    env = HTDEEnv(num_tasks=num_tasks, num_executors=num_executors,
                  complexity=complexity, max_steps=50)
    heuristic_planner = PlannerAgent(policy="heuristic")

    # obs dims: status(N) + mask(N) + features(N*2) = N*4
    ex_obs_dim  = num_tasks * 4
    ex_act_dim  = num_tasks + 1                    # task_ids + idle(-1 mapped to num_tasks)

    ppo_agents = {
        f"executor_{i}": PPOAgent(ex_obs_dim, ex_act_dim)
        for i in range(num_executors)
    }

    ep_rewards = []

    for ep in range(episodes):
        obs = env.reset()
        rollouts = {k: {"obs": [], "actions": [], "log_probs": [], "rewards": [], "dones": [], "masks": []}
                    for k in ppo_agents}
        ep_total = 0.0

        for _ in range(rollout_len):
            planner_scores = heuristic_planner.act(obs["planner"])
            actions = {"planner": planner_scores}

            for i in range(num_executors):
                key = f"executor_{i}"
                ex_obs = obs[key]
                vec  = executor_obs_vec(ex_obs)
                mask = np.append(ex_obs["available_mask"], 1)  # last action = idle

                act, lp, val = ppo_agents[key].act(vec, mask)
                task_id = act if act < num_tasks else -1
                actions[key] = task_id

                rollouts[key]["obs"].append(vec)
                rollouts[key]["actions"].append(act)
                rollouts[key]["log_probs"].append(lp)
                rollouts[key]["masks"].append(mask)

            obs, rewards, done, _ = env.step(actions)
            ep_total += sum(rewards.values())

            for i in range(num_executors):
                key = f"executor_{i}"
                rollouts[key]["rewards"].append(rewards[key])
                rollouts[key]["dones"].append(float(done))

            if done:
                obs = env.reset()

        # PPO update
        for key, agent in ppo_agents.items():
            agent.update(rollouts[key])

        ep_rewards.append(ep_total)
        if (ep + 1) % 50 == 0:
            avg = np.mean(ep_rewards[-50:])
            print(f"Episode {ep+1:4d} | avg_reward (last 50): {avg:.1f}")

    return ppo_agents, ep_rewards


if __name__ == "__main__":
    train_ppo(episodes=200)
