import numpy as np


def compute_metrics(ep_rewards: list, completion_flags: list, step_counts: list) -> dict:
    rewards = np.array(ep_rewards)
    return {
        "task_completion_rate":   float(np.mean(completion_flags)),
        "avg_reward":             float(np.mean(rewards)),
        "reward_variance":        float(np.var(rewards)),
        "convergence_speed":      _convergence_episode(rewards),
        "coordination_efficiency": float(np.mean(step_counts)),
    }


def _convergence_episode(rewards: np.ndarray, window: int = 20, threshold: float = 0.8) -> int:
    """First episode where rolling avg reward exceeds threshold * max reward."""
    max_r = rewards.max()
    for i in range(window, len(rewards)):
        if np.mean(rewards[i - window:i]) >= threshold * max_r:
            return i
    return -1  # did not converge
