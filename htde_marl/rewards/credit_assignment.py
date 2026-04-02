import numpy as np


def difference_reward(global_reward: float, agent_rewards: dict, agent_id: str) -> float:
    """
    Difference reward for agent_id:
        D_i = R_global - R_counterfactual
    Counterfactual: global reward without agent_i's contribution
    (approximated as global_reward minus agent_i's local share).
    """
    agent_local = agent_rewards.get(agent_id, 0.0)
    n = len(agent_rewards)
    avg_others = (sum(agent_rewards.values()) - agent_local) / max(n - 1, 1)
    counterfactual = global_reward - agent_local + avg_others
    return global_reward - counterfactual


def assign_credit(rewards: dict) -> dict:
    """
    Returns difference-reward-adjusted credit for each agent.
    """
    global_reward = float(np.mean(list(rewards.values())))
    return {
        agent: difference_reward(global_reward, rewards, agent)
        for agent in rewards
    }
