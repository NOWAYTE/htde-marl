# HTDE-MARL: Hierarchical Task Decomposition Environment for Multi-Agent Reinforcement Learning

A research framework for studying cooperative multi-agent RL on hierarchical task graphs. Built for publication — includes two learning algorithms, a structured reward system, and a full ablation study.

---

## Project Structure

```
htde_marl/
├── env/
│   ├── htde_env.py        # Gym-style multi-agent environment
│   ├── task_graph.py      # DAG task graph with dependency tracking
│   └── generator.py       # Random DAG generator (controllable complexity)
├── agents/
│   ├── planner.py         # Planner agent (heuristic / random)
│   ├── executor.py        # Executor agents (greedy / random)
│   └── evaluator.py       # Rule-based evaluator
├── learning/
│   ├── ppo_multiagent.py  # PPO with action masking + centralized value
│   ├── maddpg.py          # MADDPG with centralized critic
│   └── replay_buffer.py   # Experience replay buffer
├── rewards/
│   ├── reward_shaping.py  # R_global + R_local + R_coord + R_efficiency
│   └── credit_assignment.py # Difference rewards for credit assignment
├── experiments/
│   ├── train.py           # PPO training loop
│   ├── train_maddpg.py    # MADDPG training loop
│   ├── evaluate.py        # Evaluation + metrics
│   ├── compare.py         # PPO vs MADDPG comparison + plots
│   ├── ablation.py        # Ablation study (5 conditions)
│   └── configs.yaml       # All hyperparameters and experiment configs
└── utils/
    └── metrics.py         # task_completion_rate, avg_reward, convergence_speed, etc.
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib
```

---

## Run Experiments

**PPO vs MADDPG comparison:**
```bash
python htde_marl/experiments/compare.py
```
Outputs: `results/learning_curves.png`, `results/metrics_comparison.png`

**Ablation study:**
```bash
python htde_marl/experiments/ablation.py
```
Outputs: `results/ablation_curves.png`, `results/ablation_metrics.png`

---

## Environment

The `HTDEEnv` is a multi-agent environment built around a randomly generated DAG of tasks with dependency constraints.

**Agents:**
- **Planner** — outputs priority scores over all tasks each step
- **Executors (N)** — each picks one available task to execute
- **Evaluator** — rule-based critic that penalizes idle executors

**State:**
```python
{
  "adj_matrix":    (N, N),   # dependency graph
  "task_status":   (N,),     # 0=pending, 1=active, 2=done
  "task_features": (N, 2),   # [difficulty, priority]
  "available_mask":(N,),     # tasks with all deps satisfied
  "time_step":     int
}
```

**Reward structure:**
```
R_total = R_global + R_local + R_coord + R_efficiency
```
| Component | Signal |
|---|---|
| R_global | +100 on completion, -0.1/step |
| R_local | +10 valid assignment, -5 invalid |
| R_coord | +2 when all executors active |
| R_efficiency | +N for N parallel completions |

---

## Algorithms

### PPO (Multi-Agent)
- Independent actor-critic per executor
- Shared policy architecture with action masking
- Centralized value function baseline

### MADDPG
- Decentralized actors (each sees only own obs)
- Centralized critic (sees all obs + actions)
- Soft target network updates (τ=0.005)
- Gumbel noise for exploration, annealed over training

---

## Key Results

| Metric | PPO | MADDPG |
|---|---|---|
| Task completion rate | 100% | 100% |
| Avg reward (eval) | ~434 | ~444 |
| Convergence speed | ep ~20 | ep ~20 |
| Avg steps to complete | 7.9 | 8.0 |

### Ablation Study (20 tasks, 4 executors)

| Condition | Avg Reward | Δ vs Full |
|---|---|---|
| Full system | 1145.7 | — |
| No evaluator | 1124.0 | -21.7 |
| No reward shaping | 1136.6 | -9.1 |
| No coordination reward | 1135.3 | -10.4 |
| Random planner | 1134.7 | -11.0 |

---

## Credit Assignment

Implements difference rewards to isolate each agent's contribution:

```
D_i = R_global - R_counterfactual_i
```

Where the counterfactual approximates the global reward had agent *i* not acted.

---

## Extending

- **Larger graphs**: set `num_tasks=20`, `complexity=high` in `configs.yaml`
- **More executors**: set `num_executors=8`
- **Learned evaluator**: replace `EvaluatorAgent` with a trained critic network
- **GNN encoder**: replace flat obs vectors with graph embeddings in `ppo_multiagent.py`
