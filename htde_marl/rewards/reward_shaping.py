import numpy as np


def compute_rewards(
    prev_status: np.ndarray,
    curr_status: np.ndarray,
    actions: dict,
    num_executors: int,
    adj_matrix: np.ndarray,
    step: int,
    done: bool,
) -> dict:
    """
    R_total = R_global + R_local + R_coord + R_efficiency

    Returns reward dict keyed by agent name.
    """
    rewards = {f"executor_{i}": 0.0 for i in range(num_executors)}
    rewards["planner"] = 0.0

    newly_active = np.where((prev_status == 0) & (curr_status == 1))[0]
    newly_done   = np.where((prev_status != 2) & (curr_status == 2))[0]

    # --- R_local: executor rewards ---
    for i in range(num_executors):
        task_id = actions.get(f"executor_{i}", -1)
        if task_id == -1:
            rewards[f"executor_{i}"] -= 2.0                      # idle penalty (softer)
        elif task_id in newly_active or task_id in newly_done:
            rewards[f"executor_{i}"] += 10.0                     # valid assignment
        else:
            rewards[f"executor_{i}"] -= 5.0                      # invalid (already taken/done)

    # --- R_coord: bonus if no executor is idle when tasks are available ---
    available_count = int(np.sum(curr_status == 0))
    idle_executors = sum(1 for i in range(num_executors) if actions.get(f"executor_{i}", -1) == -1)
    if available_count > 0 and idle_executors == 0:
        coord_bonus = 2.0
        for k in rewards:
            rewards[k] += coord_bonus

    # --- R_efficiency: bonus for parallel execution ---
    active_this_step = len(newly_done)
    if active_this_step > 1:
        for k in rewards:
            rewards[k] += active_this_step * 1.0

    # --- R_global: step penalty + completion bonus ---
    for k in rewards:
        rewards[k] -= 0.1
    if done and np.all(curr_status == 2):
        for k in rewards:
            rewards[k] += 100.0

    # --- planner: reward balanced DAG usage (penalise bottlenecks) ---
    scores = actions.get("planner", np.zeros(len(curr_status)))
    bottlenecks = np.where(adj_matrix.sum(axis=1) > 2)[0]
    if len(bottlenecks):
        top_score_task = int(np.argmax(scores))
        rewards["planner"] += 2.0 if top_score_task in bottlenecks else -1.0

    return rewards
