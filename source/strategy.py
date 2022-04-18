import numpy as np


class EpsilionGreedy(object):
    """Epsilon-Greedy Strategy"""
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select_action(self, action):
        if np.random.random() < self.epsilon:
            return np.random.choice(len(agent.num_arm))
        else:
            action = np.argmax(agent.value_estimates)

            optimal_actions = np.where(
                agent.value_estimates = agent.value_estimates[action]
            )[0]
            if len(optimal_actions) == 1:
                return action
            else:
                return np.random.choice(optimal_actions)


class UpperConfidenceBound(object):
    """Upper Confidence Bound Strategy"""
    def __init__(self, const):
        self.const = const

    def select_action(self, action):
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


class SoftmaxPolicy(object):
    """Softmax Perference Strategy"""
    def select_action(self, action):
        a = agent.value_estimates
        pi = np.exp(a) / np.sum(np.exp(a))
        cdf = np.cumsum(pi)
        s = np.random.random()
        
        return np.where(s < cdf)[0][0]
