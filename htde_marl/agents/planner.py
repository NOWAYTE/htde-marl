import numpy as np


class PlannerAgent:
    """
    Planner agent. Outputs priority scores over all tasks.

    Policy options:
        'random'     - uniform random scores
        'heuristic'  - scores based on task features + dependency depth
                       (higher priority = more dependents + higher node priority feature)
    """

    def __init__(self, policy: str = "heuristic", seed: int = None):
        self.policy = policy
        self.rng = np.random.default_rng(seed)

    def act(self, obs: dict) -> np.ndarray:
        num_tasks = len(obs["task_status"])

        if self.policy == "random":
            return self.rng.random(num_tasks).astype(np.float32)

        # heuristic: score = priority_feature + num_dependents (downstream tasks)
        adj = obs["adj_matrix"]           # (N, N): adj[i][j]=1 means i -> j
        features = obs["task_features"]   # (N, 2): [difficulty, priority]
        status = obs["task_status"]       # (N,)

        num_dependents = adj.sum(axis=1)  # how many tasks depend on each task
        scores = features[:, 1] + num_dependents / max(num_tasks, 1)

        # zero out completed tasks
        scores[status == 2] = -1.0
        return scores.astype(np.float32)
