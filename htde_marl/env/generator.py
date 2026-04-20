import numpy as np
from htde_marl.env.task_graph import TaskGraph


def generate_task_graph(
    num_nodes: int,
    complexity_level: str = "medium",
    seed: int = None,
) -> TaskGraph:
    """
    Generate a random DAG as a TaskGraph.

    complexity_level controls edge density:
        low    -> sparse deps  (p ~ 0.2)
        medium -> moderate     (p ~ 0.4)
        high   -> dense deps   (p ~ 0.6)
    """
    rng = np.random.default_rng(seed)

    density = {"low": 0.2, "medium": 0.4, "high": 0.6}.get(complexity_level, 0.4)

    # Build a strictly upper-triangular adjacency matrix (guarantees DAG)
    adj = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if rng.random() < density:
                adj[i][j] = 1

    # Node features: [difficulty, priority] sampled uniformly in [0, 1]
    features = rng.random((num_nodes, 2)).astype(np.float32)

    graph = TaskGraph(num_nodes, node_features=features)
    graph.adj_matrix = adj
    return graph
