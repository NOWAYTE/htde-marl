import numpy as np


class ExecutorAgent:
    """
    Simple executor agent. Selects a task from available tasks.

    Policy options:
        'random'   - pick randomly from available tasks
        'greedy'   - pick highest priority task (uses planner scores if provided)
    """

    def __init__(self, agent_id: int, policy: str = "random", seed: int = None):
        self.agent_id = agent_id
        self.policy = policy
        self.rng = np.random.default_rng(seed)

    def act(self, obs: dict, planner_scores: np.ndarray = None) -> int:
        available = np.where(obs["available_mask"] == 1)[0]
        if len(available) == 0:
            return -1  # idle

        if self.policy == "greedy" and planner_scores is not None:
            return int(available[np.argmax(planner_scores[available])])

        return int(self.rng.choice(available))
