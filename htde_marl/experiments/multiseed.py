"""
Multi-seed experiment runner.
Runs PPO vs MADDPG and ablation study across multiple seeds,
then plots shaded learning curves + error bar metrics.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from htde_marl.experiments.train import train_ppo, executor_obs_vec
from htde_marl.experiments.train_maddpg import train_maddpg
from htde_marl.experiments.evaluate import run_eval
from htde_marl.experiments.ablation import _patch_rewards, _restore_rewards, eval_agents
from htde_marl.agents.planner import PlannerAgent

SEEDS      = [0, 1, 2]
EPISODES   = 400
NUM_TASKS  = 20
NUM_EXEC   = 4
EVAL_EPS   = 50
OUT_DIR    = "results/multiseed"
os.makedirs(OUT_DIR, exist_ok=True)


def smooth(x, w=20):
    return np.convolve(x, np.ones(w) / w, mode="valid")


# ── shaded curve plot ────────────────────────────────────────────────────────

def plot_shaded(curves_dict: dict, title: str, path: str):
    """curves_dict: {label: list of reward arrays (one per seed)}"""
    plt.figure(figsize=(9, 4))
    for label, seed_curves in curves_dict.items():
        smoothed = np.array([smooth(c) for c in seed_curves])
        mean = smoothed.mean(axis=0)
        std  = smoothed.std(axis=0)
        x    = np.arange(len(mean))
        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.xlabel("Episode"); plt.ylabel("Total Reward")
    plt.title(title); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")


def plot_error_bars(metrics_dict: dict, title: str, path: str):
    """metrics_dict: {label: list of metric dicts (one per seed)}"""
    keys   = ["task_completion_rate", "avg_reward", "coordination_efficiency"]
    xlabels = ["Completion Rate", "Avg Reward", "Avg Steps"]
    labels  = list(metrics_dict.keys())
    x = np.arange(len(keys))
    w = 0.8 / len(labels)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, label in enumerate(labels):
        seed_metrics = metrics_dict[label]
        means = [np.mean([m[k] for m in seed_metrics]) for k in keys]
        stds  = [np.std( [m[k] for m in seed_metrics]) for k in keys]
        ax.bar(x + i*w, means, w, yerr=stds, capsize=4, label=label)

    ax.set_xticks(x + w * (len(labels)-1) / 2)
    ax.set_xticklabels(xlabels)
    ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close()
    print(f"Saved {path}")


# ── PPO vs MADDPG multi-seed ─────────────────────────────────────────────────

def run_comparison():
    print("\n=== Multi-seed: PPO vs MADDPG ===")
    ppo_curves, maddpg_curves = [], []
    ppo_metrics, maddpg_metrics = [], []

    for seed in SEEDS:
        print(f"\n-- Seed {seed} --")
        ppo_agents, ppo_r = train_ppo(
            num_tasks=NUM_TASKS, num_executors=NUM_EXEC,
            episodes=EPISODES, seed=seed)
        ppo_curves.append(ppo_r)
        ppo_metrics.append(run_eval(ppo_agents, "ppo", NUM_TASKS, NUM_EXEC, EVAL_EPS, seed=seed+100))

        maddpg_agents, maddpg_r = train_maddpg(
            num_tasks=NUM_TASKS, num_executors=NUM_EXEC,
            episodes=EPISODES, seed=seed)
        maddpg_curves.append(maddpg_r)
        maddpg_metrics.append(run_eval(maddpg_agents, "maddpg", NUM_TASKS, NUM_EXEC, EVAL_EPS, seed=seed+100))

    plot_shaded({"PPO": ppo_curves, "MADDPG": maddpg_curves},
                "PPO vs MADDPG (mean ± std, 3 seeds)",
                f"{OUT_DIR}/learning_curves.png")
    plot_error_bars({"PPO": ppo_metrics, "MADDPG": maddpg_metrics},
                    "Evaluation Metrics (mean ± std, 3 seeds)",
                    f"{OUT_DIR}/metrics_comparison.png")

    _print_table({"PPO": ppo_metrics, "MADDPG": maddpg_metrics})


# ── Ablation multi-seed ──────────────────────────────────────────────────────

ABLATIONS = ["full", "no_evaluator", "no_shaping", "no_coord", "random_planner"]

def run_ablation():
    print("\n=== Multi-seed: Ablation Study ===")
    all_curves  = {a: [] for a in ABLATIONS}
    all_metrics = {a: [] for a in ABLATIONS}

    for seed in SEEDS:
        print(f"\n-- Seed {seed} --")
        for ablation in ABLATIONS:
            print(f"  ablation: {ablation}")
            original = _patch_rewards(ablation)
            agents, rewards = train_ppo(
                num_tasks=NUM_TASKS, num_executors=NUM_EXEC,
                episodes=EPISODES, seed=seed)
            _restore_rewards(original)

            planner = PlannerAgent(policy="random" if ablation == "random_planner" else "heuristic")
            m = eval_agents(agents, planner, NUM_TASKS, NUM_EXEC, EVAL_EPS)
            all_curves[ablation].append(rewards)
            all_metrics[ablation].append(m)

    plot_shaded(all_curves,
                "Ablation Study (mean ± std, 3 seeds)",
                f"{OUT_DIR}/ablation_curves.png")
    plot_error_bars(all_metrics,
                    "Ablation Metrics (mean ± std, 3 seeds)",
                    f"{OUT_DIR}/ablation_metrics.png")

    _print_table(all_metrics)


def _print_table(metrics_dict: dict):
    print(f"\n{'Condition':<20} {'Completion':>12} {'Avg Reward':>12} {'Avg Steps':>12}")
    print("-" * 58)
    for label, seed_metrics in metrics_dict.items():
        cr  = np.mean([m["task_completion_rate"]    for m in seed_metrics])
        ar  = np.mean([m["avg_reward"]              for m in seed_metrics])
        ar_s = np.std( [m["avg_reward"]             for m in seed_metrics])
        st  = np.mean([m["coordination_efficiency"] for m in seed_metrics])
        print(f"{label:<20} {cr:>11.0%}  {ar:>8.1f}±{ar_s:<5.1f}  {st:>8.1f}")


if __name__ == "__main__":
    run_comparison()
    run_ablation()
