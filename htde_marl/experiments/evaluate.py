import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from htde_marl.env.htde_env import HTDEEnv
from htde_marl.agents.planner import PlannerAgent
from htde_marl.experiments.train import executor_obs_vec
from htde_marl.utils.metrics import compute_metrics


def run_eval(agents, algo: str, num_tasks: int = 8, num_executors: int = 2,
             episodes: int = 50, seed: int = 99, complexity: str = "medium") -> dict:
    env = HTDEEnv(num_tasks=num_tasks, num_executors=num_executors,
                  complexity=complexity, max_steps=50, seed=seed)
    planner = PlannerAgent(policy="heuristic")
    ep_rewards, completions, steps = [], [], []

    for _ in range(episodes):
        obs = env.reset()
        total_r, t = 0.0, 0
        for t in range(50):
            scores = planner.act(obs["planner"])
            actions = {"planner": scores}
            for i in range(num_executors):
                key = f"executor_{i}"
                vec  = executor_obs_vec(obs[key])
                mask = np.append(obs[key]["available_mask"], 1)
                if algo == "ppo":
                    act, _, _ = agents[key].act(vec, mask)
                else:  # maddpg
                    act = agents[i].act(vec, mask, noise=0.0)
                actions[key] = act if act < num_tasks else -1
            obs, rewards, done, _ = env.step(actions)
            total_r += sum(rewards.values())
            if done:
                break
        ep_rewards.append(total_r)
        completions.append(env.graph.is_done())
        steps.append(t + 1)

    return compute_metrics(ep_rewards, completions, steps)
