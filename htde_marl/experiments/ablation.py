"""
Ablation study: systematically remove components and measure performance drop.

Ablations:
  - full          : all components (baseline)
  - no_evaluator  : remove evaluator signal (idle penalty zeroed)
  - no_shaping    : only global reward (+100 done, -0.1/step)
  - no_coord      : remove coordination bonus
  - random_planner: replace heuristic planner with random scores
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from htde_marl.env.htde_env import HTDEEnv
from htde_marl.agents.planner import PlannerAgent
from htde_marl.experiments.train import train_ppo, executor_obs_vec
from htde_marl.utils.metrics import compute_metrics

NUM_TASKS  = 8
NUM_EXEC   = 2
EPISODES   = 300
EVAL_EPS   = 30
OUT_DIR    = "results"
os.makedirs(OUT_DIR, exist_ok=True)


# ── reward override hooks ────────────────────────────────────────────────────

def _patch_rewards(ablation: str):
    """Monkey-patch compute_rewards based on ablation mode."""
    import htde_marl.rewards.reward_shaping as rs

    original = rs.compute_rewards

    def no_evaluator(prev, curr, actions, num_exec, adj, step, done):
        r = original(prev, curr, actions, num_exec, adj, step, done)
        # zero idle penalty (evaluator's signal)
        for i in range(num_exec):
            k = f"executor_{i}"
            if actions.get(k, -1) == -1:
                r[k] += 2.0  # undo the -2 idle penalty
        return r

    def no_shaping(prev, curr, actions, num_exec, adj, step, done):
        rewards = {f"executor_{i}": 0.0 for i in range(num_exec)}
        rewards["planner"] = 0.0
        for k in rewards:
            rewards[k] -= 0.1
        if done and np.all(curr == 2):
            for k in rewards:
                rewards[k] += 100.0
        return rewards

    def no_coord(prev, curr, actions, num_exec, adj, step, done):
        r = original(prev, curr, actions, num_exec, adj, step, done)
        # undo coord bonus: if all executors were active and tasks available
        available_count = int(np.sum(curr == 0))
        idle = sum(1 for i in range(num_exec) if actions.get(f"executor_{i}", -1) == -1)
        if available_count > 0 and idle == 0:
            for k in r:
                r[k] -= 2.0
        return r

    patches = {
        "full":           original,
        "no_evaluator":   no_evaluator,
        "no_shaping":     no_shaping,
        "no_coord":       no_coord,
        "random_planner": original,   # reward unchanged; planner swapped below
    }
    rs.compute_rewards = patches[ablation]
    return original  # return original to restore later


def _restore_rewards(original):
    import htde_marl.rewards.reward_shaping as rs
    rs.compute_rewards = original


# ── eval helper ─────────────────────────────────────────────────────────────

def eval_agents(agents, planner, num_tasks, num_exec, episodes):
    env = HTDEEnv(num_tasks=num_tasks, num_executors=num_exec,
                  complexity="medium", max_steps=50)
    ep_rewards, completions, steps = [], [], []
    for _ in range(episodes):
        obs = env.reset()
        total, t = 0.0, 0
        for t in range(50):
            scores = planner.act(obs["planner"])
            actions = {"planner": scores}
            for i in range(num_exec):
                key = f"executor_{i}"
                vec  = executor_obs_vec(obs[key])
                mask = np.append(obs[key]["available_mask"], 1)
                act, _, _ = agents[key].act(vec, mask)
                actions[key] = act if act < num_tasks else -1
            obs, rewards, done, _ = env.step(actions)
            total += sum(rewards.values())
            if done: break
        ep_rewards.append(total)
        completions.append(env.graph.is_done())
        steps.append(t + 1)
    return compute_metrics(ep_rewards, completions, steps)


# ── main ablation loop ───────────────────────────────────────────────────────

def run_ablations():
    ablations = ["full", "no_evaluator", "no_shaping", "no_coord", "random_planner"]
    results   = {}
    curves    = {}

    for ablation in ablations:
        print(f"\n=== Ablation: {ablation} ===")
        original = _patch_rewards(ablation)

        planner_policy = "random" if ablation == "random_planner" else "heuristic"
        agents, ep_rewards = train_ppo(
            num_tasks=NUM_TASKS, num_executors=NUM_EXEC,
            episodes=EPISODES, seed=0,
        )
        _restore_rewards(original)

        planner = PlannerAgent(policy=planner_policy)
        metrics = eval_agents(agents, planner, NUM_TASKS, NUM_EXEC, EVAL_EPS)
        results[ablation] = metrics
        curves[ablation]  = ep_rewards
        print(f"  completion={metrics['task_completion_rate']:.0%}  "
              f"avg_reward={metrics['avg_reward']:.1f}  "
              f"conv_ep={metrics['convergence_speed']}")

    _plot_curves(curves)
    _plot_bar(results)
    return results


def _smooth(x, w=20):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def _plot_curves(curves: dict):
    plt.figure(figsize=(9, 4))
    for label, r in curves.items():
        plt.plot(_smooth(r), label=label)
    plt.xlabel("Episode"); plt.ylabel("Total Reward (smoothed)")
    plt.title("Ablation Study: Learning Curves")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/ablation_curves.png", dpi=150)
    plt.close()
    print(f"\nSaved {OUT_DIR}/ablation_curves.png")


def _plot_bar(results: dict):
    metrics_to_plot = ["task_completion_rate", "avg_reward", "convergence_speed"]
    labels          = ["Completion Rate", "Avg Reward", "Convergence (ep)"]
    ablations       = list(results.keys())
    x = np.arange(len(metrics_to_plot))
    w = 0.15
    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, abl in enumerate(ablations):
        vals = [results[abl][m] if results[abl][m] != -1 else 0 for m in metrics_to_plot]
        ax.bar(x + idx * w, vals, w, label=abl)
    ax.set_xticks(x + w * (len(ablations) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_title("Ablation Study: Evaluation Metrics")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/ablation_metrics.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/ablation_metrics.png")


if __name__ == "__main__":
    run_ablations()
