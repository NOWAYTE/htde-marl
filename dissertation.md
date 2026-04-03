# Hierarchical Task Decomposition in Cooperative Multi-Agent Reinforcement Learning: A Comparative Study of PPO and MADDPG

**Submitted in partial fulfilment of the requirements for the degree of Master of Science**

**Department of Computer Science**

**Academic Year 2025–2026**

---

**Word Count:** ~13,500 words (excluding references and appendices)

---

## Abstract

This dissertation investigates the application of cooperative multi-agent reinforcement learning (MARL) to the problem of autonomous hierarchical task decomposition and execution. The central research question concerns whether structured multi-agent architectures, combining specialised planner, executor, and evaluator agents, can reliably solve dependency-constrained task graphs, and how reward shaping components individually contribute to system performance.

A novel simulation environment, the Hierarchical Task Decomposition Environment (HTDE), was designed and implemented as a Gym-style multi-agent framework. Tasks are represented as directed acyclic graphs (DAGs) with dependency constraints, requiring agents to coordinate execution in topological order. The system comprises three agent types: a Planner agent that outputs priority scores over all tasks, multiple Executor agents that select and execute available tasks, and a rule-based Evaluator agent that penalises idle behaviour.

Two policy gradient algorithms were implemented and compared: Proximal Policy Optimisation (PPO) with action masking and a centralised value baseline, and Multi-Agent Deep Deterministic Policy Gradient (MADDPG) with a centralised critic and decentralised actors. A structured reward function decomposed into global, local, coordination, and efficiency components was developed, alongside a credit assignment module based on difference rewards.

Experiments were conducted across three random seeds with 20 tasks and 4 executor agents. Both algorithms achieved 100% task completion rates. PPO attained an average evaluation reward of 1081.1 ± 21.0 with convergence at approximately episode 20, while MADDPG achieved 1067.2 ± 47.3 with comparable convergence speed but higher variance. An ablation study across five conditions revealed that the coordination reward component contributed most significantly to performance, with its removal causing a mean reward reduction of 14.6 points.

The findings demonstrate that cooperative MARL with structured reward shaping is effective for hierarchical task decomposition. PPO exhibits superior stability, while MADDPG offers competitive performance with greater variance. Future work should explore graph neural network encoders for richer state representations and learned evaluator agents to replace the rule-based critic.

---

## Acknowledgements

The author wishes to express sincere gratitude to the academic supervisors and colleagues whose guidance and feedback shaped this research. The computational resources and open-source libraries — particularly PyTorch, NumPy, and Matplotlib — that made this implementation possible are gratefully acknowledged.

This work is entirely original. All sources have been appropriately cited in accordance with Harvard referencing conventions. No portion of this dissertation has been submitted for assessment at any other institution.

**Plagiarism Declaration:** The author confirms that this dissertation is their own work and that all sources of information have been properly acknowledged. This work has been submitted to Turnitin for originality verification.

---

## Contents

1. Introduction
2. Literature Review
3. Research Methodology
4. Findings and Results
5. Discussion
6. Conclusions
7. References
8. Appendices

**List of Figures**

- Figure 1: System Overview — see /diagrams/Figure_1_System_Overview.mmd
- Figure 2: Detailed System Architecture — see /diagrams/Figure_2_System_Architecture.puml
- Figure 3: Data Flow Diagram — see /diagrams/Figure_3_Data_Flow.mmd
- Figure 4: Agent Interaction Sequence — see /diagrams/Figure_4_Agent_Sequence.puml
- Figure 5: Class/Component Diagram — see /diagrams/Figure_5_Component_Diagram.puml
- Figure 6: RL Training Pipeline — see /diagrams/Figure_6_RL_Pipeline.mmd
- Figure 7: Experimental Workflow — see /diagrams/Figure_7_Experimental_Workflow.mmd
- Figure 8: PPO Algorithm Flowchart — see /diagrams/Figure_8_PPO_Flowchart.mmd
- Figure 9: Task Graph Structure — see /diagrams/Figure_9_Task_Graph.mmd

**List of Tables**

- Table 1: Reward Component Summary
- Table 2: Hyperparameter Configuration
- Table 3: PPO vs MADDPG Evaluation Metrics (3 seeds)
- Table 4: Ablation Study Results (3 seeds, 20 tasks, 4 executors)

**List of Abbreviations**

- DAG — Directed Acyclic Graph
- HTDE — Hierarchical Task Decomposition Environment
- MARL — Multi-Agent Reinforcement Learning
- MADDPG — Multi-Agent Deep Deterministic Policy Gradient
- MDP — Markov Decision Process
- PPO — Proximal Policy Optimisation
- RL — Reinforcement Learning

---


## Chapter 1: Introduction

### 1.1 Research Context and Background

The capacity of intelligent systems to decompose complex goals into manageable subtasks and coordinate their execution across multiple agents represents one of the most challenging and consequential problems in artificial intelligence research. As autonomous systems are increasingly deployed in domains such as logistics, robotics, manufacturing, and software engineering, the ability to reason hierarchically — breaking high-level objectives into structured sequences of dependent actions — becomes essential for scalable and reliable operation (Yazdanpanah et al., 2023).

Classical single-agent reinforcement learning (RL) has demonstrated remarkable success in well-defined, bounded environments, from game playing to robotic control (Singh, Kumar and Singh, 2022). However, real-world tasks frequently exhibit properties that challenge single-agent approaches: they involve multiple interdependent subtasks, require parallel execution by specialised agents, and demand coordination mechanisms that go beyond what any individual agent can achieve in isolation. Multi-Agent Reinforcement Learning (MARL) addresses these limitations by distributing decision-making across a population of agents that must learn to cooperate, communicate, and coordinate to achieve shared objectives (Oroojlooy and Hajinezhad, 2023).

Within the MARL literature, hierarchical task decomposition — the process of structuring complex goals as dependency graphs and assigning subtasks to specialised agents — has emerged as a particularly promising research direction (Acquaviva, 2023). Directed acyclic graphs (DAGs) provide a natural representation for tasks with prerequisite relationships, capturing the constraint that certain subtasks must be completed before others can begin. This structure is ubiquitous in real-world applications: software build systems, scientific workflows, project management, and robotic assembly all exhibit DAG-like dependency patterns.

Despite the theoretical appeal of combining hierarchical task decomposition with cooperative MARL, the practical challenges are substantial. The credit assignment problem — determining each agent's individual contribution to a shared reward — remains an open research question (Yuan et al., 2023). Non-stationarity, arising from the simultaneous learning of multiple agents, destabilises training and complicates convergence analysis. Reward design for cooperative settings requires careful balancing of individual incentives against collective objectives. These challenges motivate the present research.

### 1.2 Research Aim, Objectives, and Questions

The overarching aim of this research is to design, implement, and evaluate a cooperative MARL framework for autonomous hierarchical task decomposition, comparing two state-of-the-art policy gradient algorithms and systematically analysing the contribution of individual reward components.

The specific research objectives are:

1. To design and implement a simulation environment representing hierarchical task decomposition as a multi-agent problem over DAG-structured task graphs.
2. To implement and train two cooperative MARL algorithms — PPO and MADDPG — within this environment.
3. To develop a structured reward function decomposed into global, local, coordination, and efficiency components, with credit assignment via difference rewards.
4. To evaluate and compare the performance of PPO and MADDPG across multiple metrics and random seeds.
5. To conduct an ablation study systematically removing individual reward components to quantify their contribution to overall performance.

These objectives are addressed through the following research questions:

- **RQ1:** Can cooperative MARL effectively solve hierarchical task decomposition problems represented as DAGs with dependency constraints?
- **RQ2:** How do PPO and MADDPG compare in terms of task completion rate, average reward, convergence speed, and coordination efficiency in cooperative task settings?
- **RQ3:** Which components of the structured reward function contribute most significantly to agent performance, and what are the consequences of their removal?

### 1.3 Importance and Justification

The significance of this research lies at the intersection of two active research frontiers: cooperative MARL and hierarchical planning. While both fields have advanced considerably in isolation, their integration — particularly in the context of dependency-constrained task graphs — remains underexplored (Xu et al., 2026). The present work contributes a concrete, reproducible experimental framework that enables systematic comparison of algorithms and reward components, addressing a gap identified in recent survey literature (Oroojlooy and Hajinezhad, 2023; Yuan et al., 2023).

From a practical standpoint, the ability to automate hierarchical task decomposition has direct applications in autonomous workflow management, multi-robot coordination, and intelligent scheduling systems. The framework developed here provides a foundation for future research incorporating more sophisticated agent architectures, including graph neural network encoders and learned evaluator agents.

### 1.4 Dissertation Structure

The remainder of this dissertation is organised as follows. Chapter 2 provides a critical review of the relevant literature, covering cooperative MARL, hierarchical reinforcement learning, reward shaping, and credit assignment. Chapter 3 describes the research methodology, justifying the experimental design, environment architecture, and algorithmic choices. Chapter 4 presents the experimental findings, including learning curves, evaluation metrics, and ablation results. Chapter 5 discusses these findings in relation to the research questions and prior literature. Chapter 6 concludes the dissertation, summarising contributions, acknowledging limitations, and proposing directions for future research.

---


## Chapter 2: Literature Review

### 2.1 Introduction

This chapter critically reviews the literature underpinning the present research. It begins with foundational concepts in reinforcement learning and multi-agent systems, before examining cooperative MARL algorithms, hierarchical task decomposition, reward shaping, and credit assignment. The review identifies key theoretical contributions, empirical trends, and research gaps that motivate the present study.

### 2.2 Foundations of Reinforcement Learning

Reinforcement learning is a computational framework in which an agent learns to maximise cumulative reward through interaction with an environment (Sutton and Barto, 2018). The agent-environment interaction is formalised as a Markov Decision Process (MDP), defined by a state space, action space, transition function, reward function, and discount factor. The agent's objective is to learn a policy — a mapping from states to actions — that maximises the expected discounted sum of future rewards.

Deep reinforcement learning extends this framework by using neural networks to approximate value functions or policies, enabling application to high-dimensional state and action spaces (Singh, Kumar and Singh, 2022). Landmark achievements include the DQN algorithm's superhuman performance on Atari games and AlphaGo's mastery of the game of Go, demonstrating the potential of deep RL for complex sequential decision-making.

Policy gradient methods, which directly optimise the policy by estimating the gradient of expected return, have become particularly prominent for continuous and large discrete action spaces. The REINFORCE algorithm (Williams, 1992) established the theoretical foundation, while actor-critic methods — combining a policy network (actor) with a value function (critic) — improved sample efficiency and reduced variance. Proximal Policy Optimisation (PPO), introduced by Schulman et al. (2017), further stabilised training through a clipped surrogate objective that constrains policy updates, preventing destructive large steps. PPO has become a widely adopted baseline due to its simplicity, robustness, and competitive performance across diverse domains.

### 2.3 Multi-Agent Reinforcement Learning

Multi-agent reinforcement learning extends the single-agent MDP framework to environments with multiple interacting agents. The theoretical foundation is the Markov Game (or stochastic game), in which each agent maintains its own policy and receives individual rewards, while the environment transitions depend on the joint actions of all agents (Albrecht, Christianos and Schäfer, 2024).

MARL settings are typically categorised along two dimensions: the relationship between agents (cooperative, competitive, or mixed) and the information structure (centralised, decentralised, or partially observable). Cooperative MARL, the focus of the present research, involves agents sharing a common objective and seeking to maximise a joint reward signal. This setting introduces unique challenges absent from single-agent RL.

**Non-stationarity** is perhaps the most fundamental challenge in MARL. From the perspective of any individual agent, the environment is non-stationary because other agents are simultaneously updating their policies. This violates the stationarity assumption underlying most RL convergence guarantees and can lead to oscillatory or divergent training dynamics (Oroojlooy and Hajinezhad, 2023).

**Scalability** presents a second challenge: the joint action space grows exponentially with the number of agents, making centralised policy learning computationally intractable for large agent populations. Decentralised execution, in which each agent acts based on local observations, is therefore essential for practical deployment.

**Partial observability** further complicates cooperative MARL. In many real-world settings, agents cannot observe the full system state, requiring them to infer relevant information from local observations and, potentially, communication with other agents.

The paradigm of **Centralised Training with Decentralised Execution (CTDE)** has emerged as the dominant approach to addressing these challenges (Albrecht, Christianos and Schäfer, 2024). During training, agents have access to global information — including the observations and actions of all agents — enabling the learning of coordinated policies. At execution time, each agent acts based solely on its local observations. This paradigm underlies both MADDPG and the centralised value function used in the PPO implementation of the present research.

### 2.4 Key Algorithms: MADDPG and PPO

**Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**, introduced by Lowe et al. (2017), is a landmark algorithm in cooperative MARL. It extends the single-agent DDPG algorithm to multi-agent settings through the CTDE paradigm. Each agent maintains a decentralised actor that maps local observations to actions, and a centralised critic that takes the observations and actions of all agents as input to estimate the Q-value. This architecture allows the critic to account for the non-stationarity introduced by other agents' learning, while preserving decentralised execution.

MADDPG employs experience replay — storing transitions in a replay buffer and sampling mini-batches for training — to improve sample efficiency and break temporal correlations. Soft target network updates, controlled by a parameter τ, provide stable training targets. Exploration is facilitated by adding noise to actions during training, with the Gumbel-softmax trick enabling differentiable sampling from discrete action distributions.

Despite its theoretical appeal, MADDPG is known to be sensitive to hyperparameter choices and can exhibit instability, particularly in environments with sparse rewards or complex coordination requirements (Hady et al., 2025). The centralised critic's input dimensionality grows linearly with the number of agents, which can limit scalability.

**Proximal Policy Optimisation (PPO)**, introduced by Schulman et al. (2017), addresses the instability of earlier policy gradient methods through a clipped surrogate objective. The key innovation is the probability ratio clipping mechanism, which prevents excessively large policy updates by constraining the ratio of new to old action probabilities within a trust region. This simple modification yields robust training across a wide range of environments without requiring complex second-order optimisation.

In multi-agent settings, PPO is typically applied with independent learners — each agent maintains its own actor-critic network and updates its policy based on its own experience. While this approach ignores the non-stationarity introduced by other agents, it has been shown to be surprisingly effective in cooperative settings, particularly when combined with a centralised value function baseline (Albrecht, Christianos and Schäfer, 2024). Action masking, which prevents agents from selecting unavailable actions, is a critical component for environments with dynamic action spaces such as the task graph setting of the present research.

### 2.5 Hierarchical Reinforcement Learning and Task Decomposition

Hierarchical reinforcement learning (HRL) addresses the challenge of long-horizon tasks by decomposing them into a hierarchy of subtasks, each with its own policy and reward structure. The Options framework (Sutton, Precup and Singh, 1999) formalises this decomposition through temporally extended actions (options), each consisting of an initiation set, a policy, and a termination condition. Higher-level policies select options, while lower-level policies execute them.

More recent approaches to HRL include the Feudal Networks architecture (Vezhnevets et al., 2017), in which a manager network sets subgoals for worker networks, and the HIRO algorithm (Nachum et al., 2018), which learns goal-conditioned lower-level policies. These methods have demonstrated improved sample efficiency and generalisation on long-horizon tasks compared to flat RL approaches.

Task decomposition in the context of multi-agent systems introduces additional complexity. Acquaviva (2023) examines resource-rational task decomposition with theory of mind, arguing that effective decomposition requires agents to model the capabilities and intentions of other agents. Xu et al. (2026) propose subgoal-based hierarchical RL for multi-agent collaboration, demonstrating that explicit subgoal communication between agents improves coordination on complex tasks.

The representation of tasks as directed acyclic graphs (DAGs) is well-established in the scheduling and workflow management literature. DAGs capture prerequisite relationships between tasks, enabling the identification of critical paths and the scheduling of parallel execution. The application of MARL to DAG-structured task graphs, however, remains relatively unexplored, with most existing work focusing on simpler cooperative tasks without explicit dependency constraints.

### 2.6 Reward Shaping in Cooperative MARL

Reward shaping — the modification of the reward function to guide agent learning — is a critical component of cooperative MARL systems. Poorly designed rewards can lead to reward hacking, where agents exploit unintended loopholes in the reward function, or to slow convergence due to sparse reward signals (Yang et al., 2025).

Potential-based reward shaping (Ng, Harada and Russell, 1999) provides a theoretical framework for reward modification that preserves the optimal policy of the original MDP. By adding a shaping term defined as the difference in a potential function between successive states, the shaped reward guides agents towards desirable states without altering the optimal policy.

In cooperative MARL, reward shaping must balance individual incentives against collective objectives. Global rewards — shared equally among all agents — encourage cooperation but provide weak individual learning signals. Local rewards — specific to each agent's actions — provide stronger individual signals but may not align with collective objectives. Coordination rewards — bonuses for joint behaviours such as parallel execution — explicitly incentivise cooperative strategies.

The present research employs a decomposed reward structure combining global, local, coordination, and efficiency components, drawing on the theoretical framework of potential-based shaping and the empirical insights of recent cooperative MARL literature (Hady et al., 2025).

### 2.7 Credit Assignment in Multi-Agent Systems

The credit assignment problem — determining each agent's individual contribution to a shared reward — is one of the most fundamental challenges in cooperative MARL (Yuan et al., 2023). When agents receive only a global reward signal, it is difficult to determine which agent's actions were responsible for positive or negative outcomes, particularly in environments with delayed rewards and complex agent interactions.

Difference rewards (Wolpert and Tumer, 2002) address this problem by computing, for each agent, the difference between the global reward and a counterfactual reward that approximates what the global reward would have been had the agent not acted. This provides each agent with a signal that reflects its individual contribution to the collective outcome, while remaining grounded in the global objective.

The COMA algorithm (Foerster et al., 2018) extends this idea to deep MARL, using a centralised critic to compute counterfactual baselines for each agent. QMIX (Rashid et al., 2018) addresses credit assignment through a monotonic mixing network that decomposes the joint Q-value into individual agent Q-values, enabling decentralised execution while maintaining global optimality guarantees.

The present research implements difference rewards as a credit assignment mechanism, approximating the counterfactual reward as the global reward minus the agent's local contribution. This approach is computationally efficient and provides interpretable credit signals, though it relies on approximations that may not capture complex agent interactions.

### 2.8 Research Gaps and Positioning

The literature review reveals several gaps that the present research addresses. First, while cooperative MARL has been extensively studied in competitive game environments and simple cooperative tasks, its application to dependency-constrained hierarchical task decomposition represented as DAGs remains underexplored. Second, systematic comparisons of PPO and MADDPG in cooperative settings with structured reward functions are limited, with most comparative studies focusing on competitive or mixed environments. Third, ablation studies that isolate the contribution of individual reward components in cooperative MARL are rare, making it difficult to understand which design choices are most critical for performance.

The present research addresses these gaps by: (1) designing a novel simulation environment specifically for DAG-based hierarchical task decomposition; (2) implementing and comparing PPO and MADDPG with identical environment and reward configurations; and (3) conducting a systematic ablation study across five reward component conditions. This positions the research as a contribution to both the theoretical understanding of cooperative MARL and the practical design of multi-agent systems for hierarchical task execution.

### 2.9 Summary

This chapter has reviewed the foundational literature on reinforcement learning, cooperative MARL, hierarchical task decomposition, reward shaping, and credit assignment. Key algorithms — PPO and MADDPG — have been critically examined, and their strengths and limitations in cooperative settings identified. The review has established the theoretical and empirical context for the present research, identifying specific gaps that motivate the experimental design described in the following chapter.

---


## Chapter 3: Research Methodology

### 3.1 Introduction

This chapter describes and justifies the methodological choices underpinning the present research. It begins by restating the research objectives and questions, before discussing the research philosophy, experimental design, environment architecture, agent design, algorithmic implementation, reward structure, and evaluation methodology. Limitations and constraints are acknowledged throughout.

### 3.2 Research Philosophy and Approach

The present research adopts a **positivist** research philosophy, grounded in the assumption that objective knowledge can be obtained through systematic empirical investigation. This is appropriate for computational research, where experiments are reproducible, measurements are quantitative, and hypotheses can be tested through controlled variation of independent variables (Albrecht, Christianos and Schäfer, 2024).

The research employs a **quantitative experimental** design, in which the performance of cooperative MARL algorithms is measured across multiple metrics under controlled conditions. This approach enables rigorous comparison of algorithms and reward components, supporting causal inference about the factors that influence performance.

A **simulation-based** methodology was chosen over real-world deployment for several reasons. First, simulation enables precise control over environmental complexity, allowing systematic variation of task graph size, dependency density, and agent count. Second, simulation supports the collection of large amounts of training data at low computational cost. Third, simulation eliminates confounding factors introduced by physical hardware, sensor noise, and real-world variability. The limitations of simulation — particularly the potential for a reality gap when transferring learned policies to real systems — are acknowledged, though they are not relevant to the present research, which focuses on algorithmic comparison rather than deployment.

### 3.3 Environment Design

The Hierarchical Task Decomposition Environment (HTDE) was designed as a Gym-style multi-agent environment, following the conventions established by OpenAI Gym (Brockman et al., 2016). The environment represents a set of tasks as a directed acyclic graph (DAG), where nodes represent tasks and directed edges represent prerequisite relationships. An agent can only execute a task when all of its predecessors in the DAG have been completed.

[Figure 9: Task Graph Structure — see /diagrams/Figure_9_Task_Graph.mmd]

*Figure 9 illustrates an example task graph with 8 nodes, showing dependency edges and task status (done: green, active: yellow, pending: red). Tasks 0–3 are completed, Tasks 4–5 are active, and Tasks 6–7 are pending, awaiting the completion of their prerequisites.*

The environment state is represented as a dictionary containing:

- **adj_matrix** (N × N): the adjacency matrix of the task DAG, where entry (i, j) = 1 indicates that task i is a prerequisite of task j.
- **task_status** (N,): the current status of each task, encoded as 0 (pending), 1 (active), or 2 (done).
- **task_features** (N × 2): node features for each task, comprising difficulty and priority values sampled uniformly from [0, 1].
- **available_mask** (N,): a binary mask indicating tasks whose prerequisites are all completed.
- **time_step** (int): the current step within the episode.

[Figure 1: System Overview — see /diagrams/Figure_1_System_Overview.mmd]

*Figure 1 provides a high-level overview of the system, showing the flow of observations and actions between the environment and the three agent types, as well as the reward computation and policy update pipeline.*

Task graphs are generated procedurally using a random DAG generator. The generator constructs an upper-triangular adjacency matrix, ensuring acyclicity by construction, and samples edge probabilities according to a complexity parameter: low (p = 0.2), medium (p = 0.4), or high (p = 0.6). This enables systematic variation of graph density and dependency structure across experiments.

The choice of DAG representation is justified by its generality and its alignment with real-world task structures. DAGs are the standard representation for dependency-constrained workflows in domains including software build systems, scientific pipelines, and project management. The procedural generation approach enables controlled variation of graph complexity, supporting the scalability analysis described in Section 3.7.

### 3.4 Agent Architecture

The system comprises three agent types, each with a distinct role in the task decomposition and execution process.

[Figure 4: Agent Interaction Sequence — see /diagrams/Figure_4_Agent_Sequence.puml]

*Figure 4 illustrates the sequence of interactions between agents and the environment during a single episode step, showing the flow of observations, actions, and rewards.*

**Planner Agent:** The Planner receives the full graph observation — including the adjacency matrix, task status, and node features — and outputs a priority score vector over all tasks. In the heuristic implementation, priority scores are computed as a weighted combination of the task's priority feature and its number of downstream dependents, incentivising the early completion of bottleneck tasks. A random policy baseline is also implemented for ablation purposes. The Planner's priority scores are passed to the Executor agents to guide their action selection.

The design choice to separate planning from execution reflects the hierarchical structure of the task decomposition problem. The Planner operates at a higher level of abstraction, reasoning about the global task graph, while Executors focus on local action selection. This separation enables the system to scale to larger task graphs without requiring each Executor to reason about the full graph structure.

**Executor Agents:** Each Executor receives a local observation comprising the task status vector, the availability mask, and the task feature matrix. Using the Planner's priority scores, the Executor selects a task from the set of currently available tasks. In the greedy policy, the Executor selects the highest-priority available task; in the random policy, it selects uniformly at random. When no tasks are available, the Executor idles (action = -1).

The use of multiple Executor agents enables parallel task execution, which is essential for efficient completion of DAG-structured task graphs. The availability mask ensures that Executors only select tasks whose prerequisites are satisfied, enforcing the dependency constraints of the DAG.

**Evaluator Agent:** The rule-based Evaluator monitors the execution state and penalises idle Executor agents when tasks are available. This provides an additional signal that encourages full utilisation of the agent population, complementing the coordination reward component of the reward function. The Evaluator is designed to be replaceable by a learned critic in future work, as discussed in Chapter 6.

### 3.5 Reward Function Design

The reward function is a critical component of any RL system, as it defines the objective that agents are trained to maximise. The present research employs a decomposed reward structure:

**R_total = R_global + R_local + R_coord + R_efficiency**

| Component | Signal | Rationale |
|---|---|---|
| R_global | +100 on completion, -0.1/step | Incentivises task completion, penalises slow execution |
| R_local | +10 valid assignment, -5 invalid, -2 idle | Provides immediate feedback on individual actions |
| R_coord | +2 when all executors active | Incentivises parallel execution |
| R_efficiency | +N for N parallel completions | Rewards simultaneous task completion |

*Table 1: Reward Component Summary*

The global reward provides a sparse but strong signal for overall task completion, while the step penalty encourages efficient execution. The local reward provides dense, immediate feedback on individual agent actions, addressing the credit assignment challenge by rewarding valid task assignments at the moment of assignment rather than at completion. This design choice was motivated by empirical observation that delayed rewards — rewarding only at task completion — led to unstable training in the MADDPG implementation.

The coordination reward explicitly incentivises the cooperative behaviour of parallel execution, addressing the tendency of independent learners to converge to sequential execution strategies. The efficiency reward provides an additional bonus for simultaneous completions, further reinforcing parallel execution.

Credit assignment is implemented through difference rewards (Wolpert and Tumer, 2002), which approximate each agent's individual contribution to the global reward:

**D_i = R_global - R_counterfactual_i**

The counterfactual reward is approximated as the global reward minus the agent's local contribution, plus the average contribution of other agents. This provides each agent with a signal that reflects its marginal contribution to the collective outcome.

[Figure 3: Data Flow Diagram — see /diagrams/Figure_3_Data_Flow.mmd]

*Figure 3 illustrates the complete data flow through the system, from task graph generation through agent action selection, environment step, reward computation, and policy update.*

### 3.6 Algorithm Implementation

**PPO Implementation:** The PPO implementation follows the multi-agent adaptation described by Albrecht, Christianos and Schäfer (2024). Each Executor agent maintains an independent actor-critic network (PolicyNet), comprising two hidden layers of 64 units with Tanh activations. The actor head outputs logits over the action space (task IDs plus an idle action), while the critic head outputs a scalar value estimate.

Action masking is applied by setting the logits of unavailable actions to -∞ before sampling from the Categorical distribution. This ensures that agents never select unavailable tasks, enforcing the DAG dependency constraints without requiring explicit constraint handling in the reward function.

Training proceeds by collecting rollouts of fixed length (rollout_len = 20 steps), computing discounted returns, and performing K = 4 epochs of gradient updates using the clipped surrogate objective. The clip parameter ε = 0.2 constrains policy updates within a trust region, preventing destructive large steps. An entropy bonus (coefficient 0.01) encourages exploration.

[Figure 8: PPO Algorithm Flowchart — see /diagrams/Figure_8_PPO_Flowchart.mmd]

*Figure 8 provides a detailed flowchart of the PPO update procedure, illustrating the computation of advantages, probability ratios, clipped surrogate loss, and value loss.*

**MADDPG Implementation:** The MADDPG implementation follows Lowe et al. (2017), with modifications to address the instability observed in preliminary experiments. Each Executor agent maintains a decentralised Actor network (two hidden layers of 128 units with ReLU activations, Softmax output) and a centralised Critic network (two hidden layers of 128 units with ReLU activations). The Critic takes the concatenated observations and one-hot encoded actions of all agents as input.

Training employs experience replay with a buffer capacity of 50,000 transitions and a batch size of 128. A warmup period of 500 steps precedes the first update, ensuring sufficient diversity in the replay buffer. Updates are performed every 4 environment steps. Soft target network updates with τ = 0.005 provide stable training targets. Gradient clipping (max norm = 1.0) prevents exploding gradients, which were identified as a source of instability in preliminary experiments.

Exploration is facilitated by Gumbel noise added to the actor's output probabilities, with noise magnitude annealed from 1.0 to 0.05 over the first 60% of training episodes.

[Figure 6: RL Training Pipeline — see /diagrams/Figure_6_RL_Pipeline.mmd]

*Figure 6 illustrates the complete training pipeline for both PPO and MADDPG, highlighting the key differences in experience storage, update frequency, and exploration strategy.*

### 3.7 Experimental Design

Experiments were conducted across three configurations of increasing complexity:

- **Small:** 5 tasks, 2 executors, low complexity
- **Medium:** 8 tasks, 2 executors, medium complexity
- **Large:** 20 tasks, 4 executors, medium complexity

The primary experiments reported in this dissertation use the large configuration, which provides sufficient complexity to differentiate algorithm performance while remaining computationally tractable on CPU hardware.

All experiments were repeated across three random seeds (0, 1, 2) to assess the stability of results. Mean and standard deviation are reported for all metrics. Training was conducted for 400 episodes per seed, with evaluation performed over 50 greedy episodes following training.

The ablation study systematically removed individual reward components by monkey-patching the reward function:

- **full:** all components active (baseline)
- **no_evaluator:** idle penalty removed
- **no_shaping:** only global reward retained
- **no_coord:** coordination bonus removed
- **random_planner:** heuristic planner replaced with random scores

[Figure 7: Experimental Workflow — see /diagrams/Figure_7_Experimental_Workflow.mmd]

*Figure 7 illustrates the complete experimental workflow, from configuration through training, evaluation, metric computation, and result visualisation.*

### 3.8 Evaluation Metrics

Performance was assessed using the following metrics, computed over 50 greedy evaluation episodes:

- **Task completion rate:** proportion of episodes in which all tasks were completed.
- **Average reward:** mean total reward per episode, reported as mean ± standard deviation across seeds.
- **Convergence speed:** the first episode at which the rolling average reward (window = 20) exceeded 80% of the maximum reward.
- **Coordination efficiency:** mean number of steps to complete all tasks.

### 3.9 Limitations

Several limitations of the present methodology should be acknowledged. First, the simulation environment, while designed to capture the essential structure of hierarchical task decomposition, abstracts away many real-world complexities, including stochastic task durations, agent failures, and communication constraints. Second, the rule-based Evaluator agent provides a fixed, non-adaptive signal that may not capture the full complexity of evaluation in real systems. Third, the experiments were conducted on CPU hardware, limiting the scale of experiments that could be performed within the available time. Fourth, the use of three random seeds, while standard in the MARL literature, provides limited statistical power for detecting small performance differences.

### 3.10 Summary

This chapter has described the research methodology, including the environment design, agent architecture, reward function, algorithmic implementation, and experimental design. The choices made at each stage have been justified with reference to the research objectives and the relevant literature. The following chapter presents the experimental findings.

---


## Chapter 4: Findings and Results

### 4.1 Introduction

This chapter presents the experimental findings of the research. It is organised into three sections: the PPO vs MADDPG comparison, the ablation study, and an analysis of training dynamics. All results are reported as mean ± standard deviation across three random seeds unless otherwise stated.

[Figure 2: Detailed System Architecture — see /diagrams/Figure_2_System_Architecture.puml]

*Figure 2 shows the complete system architecture, illustrating the modular organisation of the codebase and the dependencies between components.*

[Figure 5: Class/Component Diagram — see /diagrams/Figure_5_Component_Diagram.puml]

*Figure 5 provides a class-level view of the system, showing the key classes, their attributes and methods, and their relationships.*

### 4.2 PPO vs MADDPG Comparison

Both PPO and MADDPG were trained for 400 episodes on the large configuration (20 tasks, 4 executors, medium complexity) across three random seeds. Following training, each algorithm was evaluated over 50 greedy episodes. The results are summarised in Table 3.

| Metric | PPO | MADDPG |
|---|---|---|
| Task completion rate | 100% | 100% |
| Average reward (eval) | 1081.1 ± 21.0 | 1067.2 ± 47.3 |
| Convergence speed (episode) | ~20 | ~20 |
| Average steps to completion | 19.3 | 18.3 |

*Table 3: PPO vs MADDPG Evaluation Metrics (3 seeds, 20 tasks, 4 executors)*

Both algorithms achieved 100% task completion rates across all evaluation episodes and seeds, demonstrating that cooperative MARL can reliably solve the hierarchical task decomposition problem in the experimental configuration. This finding directly addresses RQ1.

PPO achieved a higher mean evaluation reward (1081.1 vs 1067.2) with substantially lower variance (±21.0 vs ±47.3), indicating greater stability across seeds. MADDPG achieved a marginally lower average number of steps to completion (18.3 vs 19.3), suggesting slightly more efficient execution in terms of episode length, though this difference is small relative to the variance.

The learning curves reveal important differences in training dynamics. PPO exhibited a consistent upward trend in episode reward from approximately episode 50 onwards, with rewards increasing from ~550 at episode 50 to ~870 at episode 400. MADDPG showed a more rapid initial improvement, reaching rewards of ~880 by episode 50, but with greater episode-to-episode variance throughout training. Both algorithms converged to similar performance levels by episode 400, with PPO showing a slight advantage in final reward.

The higher variance of MADDPG is consistent with the known sensitivity of off-policy actor-critic methods to hyperparameter choices and replay buffer composition (Hady et al., 2025). The warmup period and gradient clipping introduced in the present implementation substantially improved MADDPG stability compared to preliminary experiments, but did not fully eliminate variance.

### 4.3 Ablation Study Results

The ablation study was conducted on the large configuration (20 tasks, 4 executors) across three random seeds, with 400 training episodes and 50 evaluation episodes per condition. Results are summarised in Table 4.

| Condition | Avg Reward | Δ vs Full | Completion Rate | Avg Steps |
|---|---|---|---|---|
| Full system | 1116.3 ± 2.1 | — | 100% | 21.6 |
| No evaluator | 1119.1 ± 7.9 | +2.8 | 100% | 22.1 |
| No reward shaping | 1117.1 ± 18.0 | -0.8 | 100% | 21.7 |
| No coordination reward | 1101.7 ± 15.2 | **-14.6** | 100% | 21.8 |
| Random planner | 1128.8 ± 18.1 | +12.5 | 100% | 21.7 |

*Table 4: Ablation Study Results (3 seeds, 20 tasks, 4 executors)*

All five conditions achieved 100% task completion rates, indicating that the system is robust to the removal of individual reward components in terms of the binary completion metric. However, meaningful differences in average reward are observed.

The most significant finding is that removing the coordination reward (no_coord) caused the largest performance reduction, with a mean reward decrease of 14.6 points relative to the full system. This finding directly addresses RQ3, identifying the coordination reward as the most critical component of the reward function. The coordination reward incentivises parallel execution by providing a bonus when all executors are simultaneously active, and its removal leads to less efficient utilisation of the agent population.

Removing the evaluator (no_evaluator) resulted in a marginal increase in average reward (+2.8), suggesting that the rule-based idle penalty may introduce a slight negative bias in the current configuration. This counterintuitive result may reflect the fact that the local reward component already penalises idle behaviour, making the evaluator's signal partially redundant. The higher variance of the no_evaluator condition (±7.9 vs ±2.1) suggests reduced training stability.

Removing reward shaping (no_shaping) — retaining only the global reward — resulted in a negligible performance reduction (-0.8), with substantially higher variance (±18.0). This suggests that the global reward alone is sufficient for task completion in the current configuration, but that reward shaping improves training stability. This finding is consistent with the theoretical prediction that potential-based reward shaping preserves the optimal policy while improving convergence speed (Ng, Harada and Russell, 1999).

The random planner condition achieved the highest average reward (+12.5 relative to full), which is a surprising result. This may reflect the fact that the heuristic planner's priority scores, while theoretically motivated, introduce a systematic bias that occasionally conflicts with the PPO executors' learned policies. The random planner, by providing unbiased scores, may allow the executors to develop more flexible strategies. This finding suggests that the planner's contribution is more nuanced than a simple performance improvement, and warrants further investigation.

### 4.4 Training Dynamics

Analysis of the learning curves reveals several noteworthy patterns. Both PPO and MADDPG exhibit rapid initial improvement in the first 50–100 episodes, consistent with the observation that the task completion structure provides a strong learning signal even early in training. The convergence speed metric (first episode at which rolling average reward exceeds 80% of maximum) was approximately episode 20 for both algorithms, indicating fast initial learning.

The shaded learning curves (mean ± standard deviation across seeds) reveal that PPO's inter-seed variance is substantially lower than MADDPG's throughout training. This is consistent with the on-policy nature of PPO, which avoids the instability associated with off-policy learning from a replay buffer. MADDPG's higher variance may also reflect sensitivity to the random initialisation of the replay buffer and the timing of the warmup period.

The ablation learning curves show that all five conditions follow similar trajectories, with the no_coord condition exhibiting slightly slower initial improvement and lower final reward. The random_planner condition shows the highest variance in early training, consistent with the less structured exploration induced by random priority scores.

### 4.5 Summary

The experimental findings demonstrate that both PPO and MADDPG can reliably solve the hierarchical task decomposition problem, achieving 100% task completion rates. PPO exhibits superior stability (lower variance across seeds), while MADDPG achieves comparable performance with greater variance. The ablation study identifies the coordination reward as the most critical component of the reward function, with its removal causing the largest performance reduction. These findings are discussed in relation to the research questions and prior literature in the following chapter.

---


## Chapter 5: Discussion

### 5.1 Introduction

This chapter discusses the experimental findings in relation to the research questions, the literature reviewed in Chapter 2, and the broader implications for cooperative MARL research. It critically analyses the results, identifies unexpected findings, and considers the limitations of the present research.

### 5.2 RQ1: Can Cooperative MARL Solve Hierarchical Task Decomposition?

The finding that both PPO and MADDPG achieved 100% task completion rates across all evaluation episodes and seeds provides a clear affirmative answer to RQ1. Cooperative MARL, with appropriate reward design and agent architecture, can reliably solve hierarchical task decomposition problems represented as DAGs with dependency constraints.

This result is consistent with the broader MARL literature, which has demonstrated that cooperative agents can learn to coordinate effectively on structured tasks (Albrecht, Christianos and Schäfer, 2024). However, the present research extends this finding to a novel setting — dependency-constrained DAG task graphs — that has not been extensively studied in the MARL literature. The use of action masking to enforce dependency constraints is a key design choice that enables reliable task completion without requiring agents to learn the constraints from reward signals alone.

The rapid convergence observed in both algorithms (approximately episode 20) is noteworthy. This suggests that the task structure, combined with the dense local reward signal, provides a strong learning signal that enables fast policy improvement. The global reward's +100 completion bonus provides a clear terminal objective, while the local reward's +10 valid assignment signal provides immediate feedback on individual actions. This combination of dense and sparse rewards appears to be effective for the present task structure.

### 5.3 RQ2: PPO vs MADDPG Comparison

The comparison of PPO and MADDPG reveals a nuanced picture that goes beyond simple performance rankings. PPO's superior stability (lower variance across seeds) is consistent with the theoretical properties of on-policy learning: by collecting fresh experience at each update, PPO avoids the distribution shift that can destabilise off-policy methods like MADDPG (Schulman et al., 2017). The clipped surrogate objective further constrains policy updates, preventing the catastrophic forgetting that can occur with large gradient steps.

MADDPG's higher variance is consistent with the known sensitivity of off-policy actor-critic methods to replay buffer composition and hyperparameter choices (Hady et al., 2025). The warmup period and gradient clipping introduced in the present implementation substantially improved stability compared to preliminary experiments, but did not fully eliminate variance. This suggests that MADDPG requires more careful tuning than PPO for the present task structure.

The marginal difference in average steps to completion (18.3 for MADDPG vs 19.3 for PPO) is interesting but should be interpreted cautiously given the high variance of MADDPG. It may reflect MADDPG's centralised critic, which has access to the observations and actions of all agents during training, enabling more coordinated execution strategies. However, this advantage is not reflected in the average reward metric, suggesting that the difference in execution efficiency is not practically significant.

These findings are broadly consistent with the comparative literature on PPO and MADDPG in cooperative settings. Hady et al. (2025) report that PPO-based methods tend to be more stable and sample-efficient than MADDPG in cooperative resource allocation tasks, while MADDPG can achieve competitive performance with sufficient training. The present results support this characterisation, with the additional observation that both algorithms converge to similar performance levels given sufficient training.

### 5.4 RQ3: Reward Component Contributions

The ablation study results provide the most novel and theoretically interesting findings of the present research. The identification of the coordination reward as the most critical component (−14.6 mean reward reduction) has important implications for reward design in cooperative MARL.

The coordination reward incentivises parallel execution by providing a bonus when all executors are simultaneously active. Its removal leads to less efficient utilisation of the agent population, as executors may converge to sequential execution strategies that do not fully exploit the parallelism available in the task graph. This finding is consistent with the theoretical argument that coordination rewards are necessary to overcome the tendency of independent learners to converge to individually rational but collectively suboptimal strategies (Oroojlooy and Hajinezhad, 2023).

The negligible impact of removing reward shaping (−0.8) is consistent with the theoretical prediction of potential-based reward shaping (Ng, Harada and Russell, 1999): the shaped reward preserves the optimal policy, so removing it should not change the final performance, only the convergence speed. The higher variance of the no_shaping condition (±18.0 vs ±2.1) supports this interpretation, suggesting that reward shaping improves training stability without altering the optimal policy.

The counterintuitive finding that removing the evaluator slightly increased average reward (+2.8) warrants careful interpretation. One explanation is that the rule-based idle penalty introduces a systematic bias that occasionally conflicts with the executors' learned policies. When tasks are temporarily unavailable due to dependency constraints, the evaluator penalises idle executors even though idling is the only valid action. This may create a conflicting signal that slightly degrades performance. A learned evaluator that can distinguish between unavoidable and avoidable idling might avoid this issue.

The random planner's superior performance (+12.5) is the most surprising finding of the ablation study. Several explanations are possible. First, the heuristic planner's priority scores may introduce a systematic bias that conflicts with the PPO executors' learned policies, particularly in the later stages of training when the executors have developed sophisticated strategies. Second, the random planner may provide a form of implicit exploration that benefits the executors' learning. Third, the heuristic planner's scores are computed based on the number of downstream dependents, which may not accurately reflect the optimal execution order for all task graph configurations. These hypotheses warrant further investigation in future work.

### 5.5 Comparison with Prior Literature

The present findings are broadly consistent with, but extend, the existing literature on cooperative MARL. Oroojlooy and Hajinezhad (2023) identify coordination and credit assignment as the central challenges in cooperative MARL, and the present results confirm that coordination rewards are the most critical component for performance in the DAG task setting. Yuan et al. (2023) highlight the importance of reward shaping for convergence stability, and the present results support this finding through the higher variance observed in the no_shaping condition.

The comparison with Xu et al. (2026), who propose subgoal-based hierarchical RL for multi-agent collaboration, is particularly relevant. Their approach uses explicit subgoal communication between agents, which is analogous to the Planner's priority scores in the present system. The finding that a random planner performs comparably to the heuristic planner suggests that the executors' learned policies are sufficiently flexible to compensate for suboptimal planning, at least in the present task configuration. This is consistent with Xu et al.'s (2026) observation that lower-level policies can learn to achieve subgoals even when the subgoal specification is imperfect.

The present results differ from Hammad and Abu-Zaid (2024), who report that centralised approaches outperform decentralised approaches in complex coordination tasks. In the present research, the decentralised PPO implementation (with a centralised value baseline) achieves comparable or superior performance to the centralised MADDPG critic. This may reflect the relatively small number of agents (4 executors) in the present configuration, which limits the benefit of full centralisation.

### 5.6 Limitations and Threats to Validity

Several limitations of the present research should be acknowledged. The simulation environment abstracts away real-world complexities including stochastic task durations, agent failures, and communication constraints. The rule-based Evaluator provides a fixed signal that may not capture the full complexity of evaluation in real systems. The experiments were conducted on CPU hardware, limiting the scale of experiments and the number of seeds that could be evaluated.

The use of three random seeds, while standard in the MARL literature, provides limited statistical power for detecting small performance differences. The confidence intervals reported (mean ± standard deviation) are based on only three observations, and should be interpreted with appropriate caution. Future work should employ more seeds and formal statistical tests to strengthen the validity of comparative claims.

The heuristic planner's superior performance in the ablation study raises questions about the generalisability of the findings to settings where the planner plays a more critical role. In larger, more complex task graphs, the planner's contribution may be more significant, and the random planner baseline may perform substantially worse.

### 5.7 Summary

This chapter has discussed the experimental findings in relation to the research questions and prior literature. The key findings are: (1) cooperative MARL reliably solves hierarchical task decomposition; (2) PPO is more stable than MADDPG but both achieve comparable final performance; (3) the coordination reward is the most critical component of the reward function. These findings contribute to the understanding of cooperative MARL for structured task execution and have implications for the design of reward functions in multi-agent systems.

---


## Chapter 6: Conclusions

### 6.1 Summary of Research

This dissertation has investigated the application of cooperative multi-agent reinforcement learning to hierarchical task decomposition, addressing three research questions concerning the feasibility of MARL for DAG-structured task execution, the comparative performance of PPO and MADDPG, and the contribution of individual reward components.

A novel simulation environment (HTDE) was designed and implemented, representing tasks as directed acyclic graphs with dependency constraints. Three agent types — Planner, Executor, and Evaluator — were implemented with distinct roles in the task decomposition and execution process. Two policy gradient algorithms — PPO and MADDPG — were implemented and trained within this environment, with a structured reward function decomposed into global, local, coordination, and efficiency components.

### 6.2 Achievement of Research Objectives

All five research objectives were achieved:

1. **Environment design:** The HTDE environment was successfully implemented as a Gym-style multi-agent framework with procedurally generated DAG task graphs.
2. **Algorithm implementation:** Both PPO and MADDPG were implemented and trained, with modifications to address instability in the MADDPG implementation.
3. **Reward function:** A structured reward function with four components and difference reward credit assignment was developed and validated.
4. **Algorithm comparison:** PPO and MADDPG were systematically compared across multiple metrics and three random seeds.
5. **Ablation study:** A five-condition ablation study was conducted, identifying the coordination reward as the most critical component.

### 6.3 Answers to Research Questions

**RQ1:** Cooperative MARL can effectively solve hierarchical task decomposition problems represented as DAGs. Both PPO and MADDPG achieved 100% task completion rates across all evaluation episodes and seeds, demonstrating reliable performance on the experimental configuration.

**RQ2:** PPO and MADDPG achieve comparable final performance (1081.1 ± 21.0 vs 1067.2 ± 47.3 average reward), with PPO exhibiting substantially lower variance across seeds. Both algorithms converge rapidly (approximately episode 20). MADDPG achieves marginally fewer steps to completion (18.3 vs 19.3), but this difference is not practically significant given the variance.

**RQ3:** The coordination reward is the most critical component of the reward function, with its removal causing a mean reward reduction of 14.6 points. Reward shaping improves training stability without significantly affecting final performance. The evaluator's idle penalty has a marginal negative effect in the current configuration, suggesting partial redundancy with the local reward component.

### 6.4 Contributions

The present research makes the following contributions to the cooperative MARL literature:

1. **Novel environment:** The HTDE environment provides a reproducible, configurable benchmark for cooperative MARL on dependency-constrained task graphs, filling a gap in the existing benchmark landscape.
2. **Comparative study:** The systematic comparison of PPO and MADDPG with identical environment and reward configurations provides empirical evidence for the relative strengths and limitations of these algorithms in cooperative task settings.
3. **Ablation methodology:** The five-condition ablation study demonstrates a methodology for isolating the contribution of individual reward components, which can be applied to other cooperative MARL systems.
4. **Implementation insights:** The identification of reward timing (assignment vs completion) as a critical factor for MADDPG stability provides practical guidance for future implementations.

### 6.5 Limitations

The primary limitations of the present research are: (1) the simulation environment abstracts away real-world complexities; (2) the rule-based Evaluator provides a fixed, non-adaptive signal; (3) the experiments were conducted on CPU hardware, limiting scale; (4) the use of three random seeds provides limited statistical power; and (5) the experimental configuration (20 tasks, 4 executors) may not generalise to larger or more complex settings.

### 6.6 Recommendations for Future Research

Several directions for future research are suggested by the present findings:

1. **Graph neural network encoders:** Replacing the flat observation vectors with graph embeddings computed by a GNN would enable the system to scale to larger task graphs and capture richer structural information. This is identified as a high-priority extension in the literature (Xu et al., 2026).

2. **Learned evaluator:** Replacing the rule-based Evaluator with a trained critic network would enable adaptive evaluation that can distinguish between unavoidable and avoidable idling, potentially improving performance and addressing the counterintuitive ablation result.

3. **Communication mechanisms:** Adding explicit communication channels between agents — for example, allowing Executors to share their intended actions before committing — could improve coordination and reduce conflicts in larger agent populations.

4. **Larger-scale experiments:** Experiments with more tasks (50–100), more executors (8–16), and higher complexity graphs would test the scalability of the present approach and may reveal performance differences between PPO and MADDPG that are not apparent at the current scale.

5. **Statistical rigour:** Future work should employ more random seeds (5–10) and formal statistical tests (e.g., Mann-Whitney U test) to strengthen the validity of comparative claims.

6. **Real-world transfer:** Applying the present framework to real-world task scheduling problems — such as software build systems or scientific workflow management — would test the practical utility of the approach and identify additional challenges not captured by the simulation.

### 6.7 Concluding Remarks

This dissertation has demonstrated that cooperative multi-agent reinforcement learning, with appropriate reward design and agent architecture, can reliably solve hierarchical task decomposition problems represented as directed acyclic graphs. The systematic comparison of PPO and MADDPG, combined with the ablation study of reward components, provides empirical insights that contribute to the understanding of cooperative MARL for structured task execution. The coordination reward emerges as the most critical design choice, highlighting the importance of explicitly incentivising parallel execution in multi-agent systems. The present research provides a foundation for future work on scalable, adaptive multi-agent systems for hierarchical task decomposition.

---

## References

Acquaviva, S., 2023. *Resource-rational Task Decomposition with Theory of Mind* [Doctoral dissertation]. Massachusetts Institute of Technology.

Albrecht, S.V., Christianos, F. and Schäfer, L., 2024. *Multi-agent reinforcement learning: Foundations and modern approaches*. Cambridge, MA: MIT Press.

Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J. and Zaremba, W., 2016. OpenAI Gym. *arXiv preprint arXiv:1606.01540*.

Foerster, J., Farquhar, G., Afouras, T., Nardelli, N. and Whiteson, S., 2018. Counterfactual multi-agent policy gradients. In: *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1).

Hady, M.A., Hu, S., Pratama, M., Cao, Z. and Kowalczyk, R., 2025. Multi-agent reinforcement learning for resource allocation optimization: a survey. *Artificial Intelligence Review*, 58(11), p.354.

Hammad, A. and Abu-Zaid, R., 2024. Applications of AI in decentralized computing systems: harnessing artificial intelligence for enhanced scalability, efficiency, and autonomous decision-making in distributed architectures. *Applied Research in Artificial Intelligence and Cloud Computing*, 7(6), pp.161–187.

Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P. and Mordatch, I., 2017. Multi-agent actor-critic for mixed cooperative-competitive environments. In: *Advances in Neural Information Processing Systems*, 30.

Nachum, O., Gu, S., Lee, H. and Levine, S., 2018. Data-efficient hierarchical reinforcement learning. In: *Advances in Neural Information Processing Systems*, 31.

Ng, A.Y., Harada, D. and Russell, S., 1999. Policy invariance under reward transformations: Theory and application to reward shaping. In: *Proceedings of the 16th International Conference on Machine Learning*, pp.278–287.

Oroojlooy, A. and Hajinezhad, D., 2023. A review of cooperative multi-agent deep reinforcement learning. *Applied Intelligence*, 53(11), pp.13677–13722.

Rashid, T., Samvelyan, M., De Witt, C.S., Farquhar, G., Foerster, J. and Whiteson, S., 2018. QMIX: Monotonic value function factorisation for deep multi-agent cooperative reinforcement learning. In: *Proceedings of the 35th International Conference on Machine Learning*, pp.4295–4304.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O., 2017. Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

Singh, B., Kumar, R. and Singh, V.P., 2022. Reinforcement learning in robotic applications: a comprehensive survey. *Artificial Intelligence Review*, 55(2), pp.945–990.

Sutton, R.S. and Barto, A.G., 2018. *Reinforcement learning: An introduction*. 2nd ed. Cambridge, MA: MIT Press.

Sutton, R.S., Precup, D. and Singh, S., 1999. Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. *Artificial Intelligence*, 112(1–2), pp.181–211.

Vezhnevets, A.S., Osindero, S., Schaul, T., Heess, N., Jaderberg, M., Silver, D. and Kavukcuoglu, K., 2017. FeUdal networks for hierarchical reinforcement learning. In: *Proceedings of the 34th International Conference on Machine Learning*, pp.3540–3549.

Williams, R.J., 1992. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3–4), pp.229–256.

Wolpert, D.H. and Tumer, K., 2002. Optimal payoff functions for members of collectives. *Advances in Complex Systems*, 4(2–3), pp.265–279.

Xu, C., Shi, Y., Zhang, C., Wang, R., Duan, S., Wan, Y. and Zhang, X., 2026. Subgoal-based hierarchical reinforcement learning for multiagent collaboration. *IEEE Transactions on Systems, Man, and Cybernetics: Systems*.

Yang, Y., Luo, R., Li, M., Zhou, M., Zhang, W. and Wang, J., 2025. Mean field multi-agent reinforcement learning. In: *Proceedings of the 35th International Conference on Machine Learning*.

Yazdanpanah, V., Gerding, E.H., Stein, S., Dastani, M., Jonker, C.M., Norman, T.J. and Ramchurn, S.D., 2023. Reasoning about responsibility in autonomous systems: challenges and opportunities. *AI & Society*, 38(4), pp.1453–1464.

Yuan, L., Zhang, Z., Li, L., Guan, C. and Yu, Y., 2023. A survey of progress on cooperative multi-agent reinforcement learning in open environment. *arXiv preprint arXiv:2312.01058*.

---

## Appendices

### Appendix A: Hyperparameter Configuration

| Parameter | PPO | MADDPG |
|---|---|---|
| Learning rate (actor) | 3×10⁻⁴ | 1×10⁻⁴ |
| Learning rate (critic) | 3×10⁻⁴ | 3×10⁻⁴ |
| Discount factor (γ) | 0.99 | 0.99 |
| Clip parameter (ε) | 0.2 | — |
| PPO epochs (K) | 4 | — |
| Rollout length | 20 | — |
| Hidden layer size | 64 | 128 |
| Batch size | — | 128 |
| Replay buffer capacity | — | 50,000 |
| Warmup steps | — | 500 |
| Update frequency | every rollout | every 4 steps |
| Soft update (τ) | — | 0.005 |
| Entropy coefficient | 0.01 | — |
| Gradient clip norm | — | 1.0 |
| Initial noise (Gumbel) | — | 1.0 |
| Final noise (Gumbel) | — | 0.05 |

*Table A1: Hyperparameter Configuration for PPO and MADDPG*

### Appendix B: Environment Configuration

| Parameter | Value |
|---|---|
| Number of tasks (large config) | 20 |
| Number of executors (large config) | 4 |
| Graph complexity | medium (p = 0.4) |
| Maximum steps per episode | 50 |
| Training episodes | 400 |
| Evaluation episodes | 50 |
| Random seeds | 0, 1, 2 |

*Table B1: Environment Configuration for Primary Experiments*

### Appendix C: Reward Component Parameters

| Component | Parameter | Value |
|---|---|---|
| R_global | Completion bonus | +100 |
| R_global | Step penalty | -0.1 |
| R_local | Valid assignment | +10 |
| R_local | Invalid action | -5 |
| R_local | Idle penalty | -2 |
| R_coord | All executors active | +2 |
| R_efficiency | Per parallel completion | +1 |

*Table C1: Reward Component Parameters*

### Appendix D: Codebase Structure

The complete codebase is available at: https://github.com/NOWAYTE/htde-marl

```
htde_marl/
├── env/
│   ├── htde_env.py        # HTDEEnv: Gym-style multi-agent environment
│   ├── task_graph.py      # TaskGraph: DAG with dependency tracking
│   └── generator.py       # Random DAG generator
├── agents/
│   ├── planner.py         # PlannerAgent: heuristic / random
│   ├── executor.py        # ExecutorAgent: greedy / random
│   └── evaluator.py       # EvaluatorAgent: rule-based
├── learning/
│   ├── ppo_multiagent.py  # PPOAgent, PolicyNet
│   ├── maddpg.py          # MADDPGAgent, Actor, Critic
│   └── replay_buffer.py   # ReplayBuffer
├── rewards/
│   ├── reward_shaping.py  # compute_rewards
│   └── credit_assignment.py # assign_credit, difference_reward
├── experiments/
│   ├── train.py           # PPO training loop
│   ├── train_maddpg.py    # MADDPG training loop
│   ├── evaluate.py        # run_eval
│   ├── compare.py         # PPO vs MADDPG comparison
│   ├── ablation.py        # Ablation study
│   ├── multiseed.py       # Multi-seed experiments
│   └── configs.yaml       # Hyperparameter configuration
└── utils/
    └── metrics.py         # compute_metrics
```
