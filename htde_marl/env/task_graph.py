import numpy as np


class TaskGraph:
    def __init__(self, num_nodes: int, node_features: np.ndarray = None):
        """
        num_nodes: number of tasks in the DAG
        node_features: (N, F) array of [difficulty, priority] per task
        """
        self.num_nodes = num_nodes
        self.adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)  # adj_matrix[i][j] = 1 means i -> j (i is prereq of j)
        self.node_features = node_features if node_features is not None else np.ones((num_nodes, 2))
        self.status = np.zeros(num_nodes, dtype=int)  # 0=pending, 1=active, 2=done

    def get_available_tasks(self) -> list[int]:
        """Return task IDs where all dependencies are completed and task is still pending."""
        available = []
        for task in range(self.num_nodes):
            if self.status[task] != 0:
                continue
            deps = np.where(self.adj_matrix[:, task] == 1)[0]
            if all(self.status[d] == 2 for d in deps):
                available.append(task)
        return available

    def mark_active(self, task_id: int):
        assert self.status[task_id] == 0, f"Task {task_id} is not pending"
        self.status[task_id] = 1

    def mark_completed(self, task_id: int):
        assert self.status[task_id] in (0, 1), f"Task {task_id} already done"
        self.status[task_id] = 2

    def is_done(self) -> bool:
        return bool(np.all(self.status == 2))

    def reset(self):
        self.status = np.zeros(self.num_nodes, dtype=int)
