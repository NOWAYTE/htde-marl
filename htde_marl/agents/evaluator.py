import numpy as np


class EvaluatorAgent:
    """
    Rule-based evaluator. Scores the current state and returns a shaped reward signal.
    Option 2 (learned critic) will replace this later.
    """

    def evaluate(self, obs: dict, actions: dict, num_executors: int) -> float:
        status = obs["task_status"]
        done_ratio = np.sum(status == 2) / len(status)

        # penalise idle executors when tasks are available
        available = np.sum(obs["available_mask"]) if "available_mask" in obs else 0
        idle = sum(1 for i in range(num_executors) if actions.get(f"executor_{i}", -1) == -1)
        idle_penalty = -idle * 1.0 if available > 0 else 0.0

        return float(done_ratio * 5.0 + idle_penalty)
