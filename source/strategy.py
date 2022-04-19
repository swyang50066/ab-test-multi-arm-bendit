import numpy as np


class EpsilonGreedy(object):
    """Epsilon-Greedy Strategy"""
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select_action(self, agent):
        if np.random.random() < self.epsilon:
            return np.random.choice(agent.num_arm)
        else:
            action = np.argmax(agent.value_estimates)

        optimal_actions = np.where(
            agent.value_estimates == agent.value_estimates[action]
        )[0]
        if len(optimal_actions) == 1:
            return action
        else:
            return np.random.choice(optimal_actions)


class UpperConfidenceBound(object):
    """Upper Confidence Bound Strategy"""
    def __init__(self, const):
        self.const = const

    def select_action(self, agent):
        exploration = np.log(agent.trial+1) / agent.action_counts
        exploration[np.isnan(exploration)] = 0
        exploration = np.power(exploration, 1/self.const)

        q = agent.value_estimates + exploration
        
        action = np.argmax(q)
        optimal_actions = np.where(q == q[action])[0]
        if len(optimal_actions) == 1:
            return action
        else:
            return np.random.choice(optimal_actions)


class Softmax(object):
    """Softmax Perference Strategy"""
    def select_action(self, agent):
        pi = (
            np.exp(agent.value_estimates) 
            / np.sum(np.exp(agent.value_estimates))
        )
        
        return np.where(
            np.random.random() < np.cumsum(pi)
        )[0][0]
