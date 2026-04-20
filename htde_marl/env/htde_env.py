import numpy as np
from htde_marl.env.generator import generate_task_graph
from htde_marl.rewards.reward_shaping import compute_rewards
from htde_marl.rewards.credit_assignment import assign_credit


class HTDEEnv:
    """
    Hierarchical Task Decomposition Environment (Multi-Agent).

    Agents:
        - 1 planner
        - num_executors executors
        - 1 evaluator (rule-based for now)

    Step contract:
        actions = {
            "planner":    np.ndarray (N,)   priority scores over tasks
            "executor_i": int               task_id to execute (or -1 to idle)
        }
        returns: obs, rewards, done, info
    """

    def __init__(self, num_tasks: int = 10, num_executors: int = 2,
                 complexity: str = "medium", max_steps: int = 50, seed: int = None):
        self.num_tasks = num_tasks
        self.num_executors = num_executors
        self.complexity = complexity
        self.max_steps = max_steps
        self.seed = seed

        self.graph = None
        self.step_count = 0
        self._active_assignments: dict[int, int] = {}  # executor_id -> task_id

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        self.graph = generate_task_graph(self.num_tasks, self.complexity, self.seed)
        self.step_count = 0
        self._active_assignments = {}
        return self._get_obs()

    def step(self, actions: dict) -> tuple[dict, dict, bool, dict]:
        assert self.graph is not None, "Call reset() first"

        available = set(self.graph.get_available_tasks())
        prev_status = self.graph.status.copy()

        # --- complete tasks that were active last step ---
        for ex_id, task_id in list(self._active_assignments.items()):
            self.graph.mark_completed(task_id)
        self._active_assignments = {}

        # --- assign new tasks from executor actions ---
        claimed = set()
        for i in range(self.num_executors):
            task_id = actions.get(f"executor_{i}", -1)
            if task_id != -1 and task_id in available and task_id not in claimed:
                self.graph.mark_active(task_id)
                self._active_assignments[i] = task_id
                claimed.add(task_id)

        curr_status = self.graph.status.copy()
        done = self.graph.is_done() or self.step_count >= self.max_steps

        rewards = compute_rewards(
            prev_status, curr_status, actions,
            self.num_executors, self.graph.adj_matrix,
            self.step_count, done,
        )

        self.step_count += 1
        new_available = self.graph.get_available_tasks()
        obs = self._get_obs()
        info = {"step": self.step_count, "available_tasks": new_available}
        return obs, rewards, done, info

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict:
        g = self.graph
        obs = {
            "planner": {
                "adj_matrix":    g.adj_matrix.copy(),
                "task_status":   g.status.copy(),
                "task_features": g.node_features.copy(),
                "time_step":     self.step_count,
            }
        }
        available_mask = np.zeros(self.num_tasks, dtype=int)
        for t in g.get_available_tasks():
            available_mask[t] = 1

        for i in range(self.num_executors):
            obs[f"executor_{i}"] = {
                "task_status":    g.status.copy(),
                "available_mask": available_mask.copy(),
                "task_features":  g.node_features.copy(),
                "assigned_task":  self._active_assignments.get(i, -1),
            }

        obs["evaluator"] = {
            "task_status": g.status.copy(),
            "step":        self.step_count,
        }
        return obs
