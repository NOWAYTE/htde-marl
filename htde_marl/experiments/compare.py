import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from htde_marl.experiments.train import train_ppo
from htde_marl.experiments.train_maddpg import train_maddpg
from htde_marl.experiments.evaluate import run_eval

EPISODES = 400
NUM_TASKS = 8
NUM_EXEC  = 2
OUT_DIR   = "results"
os.makedirs(OUT_DIR, exist_ok=True)


def smooth(x, w=20):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_learning_curves(ppo_r, maddpg_r):
    plt.figure(figsize=(8, 4))
    plt.plot(smooth(ppo_r),    label="PPO",    color="steelblue")
    plt.plot(smooth(maddpg_r), label="MADDPG", color="darkorange")
    plt.xlabel("Episode"); plt.ylabel("Total Reward (smoothed)")
    plt.title("Learning Curves: PPO vs MADDPG")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/learning_curves.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/learning_curves.png")


def plot_metrics_bar(ppo_m, maddpg_m):
    keys   = ["task_completion_rate", "avg_reward", "coordination_efficiency"]
    labels = ["Completion Rate", "Avg Reward", "Avg Steps to Done"]
    x = np.arange(len(keys))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, [ppo_m[k]    for k in keys], w, label="PPO",    color="steelblue")
    ax.bar(x + w/2, [maddpg_m[k] for k in keys], w, label="MADDPG", color="darkorange")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Evaluation Metrics: PPO vs MADDPG")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/metrics_comparison.png", dpi=150)
    plt.close()
    print(f"Saved {OUT_DIR}/metrics_comparison.png")


if __name__ == "__main__":
    print("=== Training PPO ===")
    ppo_agents, ppo_rewards = train_ppo(
        num_tasks=NUM_TASKS, num_executors=NUM_EXEC, episodes=EPISODES, seed=0)

    print("\n=== Training MADDPG ===")
    maddpg_agents, maddpg_rewards = train_maddpg(
        num_tasks=NUM_TASKS, num_executors=NUM_EXEC, episodes=EPISODES, seed=0)

    print("\n=== Evaluating ===")
    ppo_m    = run_eval(ppo_agents,    "ppo",    NUM_TASKS, NUM_EXEC)
    maddpg_m = run_eval(maddpg_agents, "maddpg", NUM_TASKS, NUM_EXEC)

    print("\nPPO    metrics:", {k: round(v, 2) for k, v in ppo_m.items()})
    print("MADDPG metrics:", {k: round(v, 2) for k, v in maddpg_m.items()})

    plot_learning_curves(ppo_rewards, maddpg_rewards)
    plot_metrics_bar(ppo_m, maddpg_m)
