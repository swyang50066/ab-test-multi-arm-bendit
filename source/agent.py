import numpy as np


class Agent(object):
    """Agent Wrapper Class"""
    def __init__(self, policy, num_arm, prior=0, gamma=None):
        # Policy
        self.policy = policy
        
        # Parameters
        self.num_arm = num_arm
        self.prior = prior
        self.gamma = gamma
        
        # Value spaces
        self.value_estimates = self.prior*np.ones(self.num_arm)
        self.action_counts = np.zeros(self.num_arm, dtype=np.int16)

        # Histories
        self.action_history = list()
        self.regret_history = list()

        self.trial = 0

    def __str__(self):
        return "Normal Agent"

    def select_action(self):
        """Select action under policy"""
        selected_action = self.policy.select_action(self)
        self.action_counts[selected_action] += 1
        self.action_history.append(selected_action)
        self.trial+=1

        return selected_action

    def observe(self, reward):
        """Observe environment"""
        last_action = self.action_history[-1]
        
        if self.gamma is None:
            gamma = float(1 / self.action_counts[last_action])
        else:
            gamma = self.gamma

        self.value_estimates[last_action] += (
            gamma*(reward - self.value_estimates[last_action])
        )

    def reset(self):
        """Reset memory"""
        self.value_estimates = self.prior*np.ones(self.num_arm)
        self.action_counts = np.zeros(self.num_arm, dtype=np.int16)

        # Histories
        self.action_history = list()
        self.regret_history = list()

        self.trial = 0


class GradientAgent(Agent):
    """Agent class with an approach based on a preference quantity"""
    def __init__(
        self, 
        policy, 
        num_arm, 
        prior=0, 
        alpha=0.1, 
        b_use_baseline=True
    ):
        super(GradientAgent, self).__init__(
            policy=policy, num_arm=num_arm, prior=prior
        )

        # Parameters
        self.alpha = alpha
        self.b_use_baseline = b_use_baseline
        self.averaged_reward = 0

    def __str__(self):
        return "Gradient Agent"


    def observe(self, reward):
        """Observe environment"""
        last_action = self.action_history[-1]
        
        if self.b_use_baseline:
            delta = reward - self.averaged_reward
            self.averaged_reward += delta/np.sum(self.action_counts)

        pi = (
            np.exp(self.value_estimates) 
            / np.sum(np.exp(self.value_estimates))
        )

        ht = self.value_estimates[last_action]
        ht += (
            self.alpha*(reward - self.averaged_reward)*(1 - pi[last_action])
        )
        
        self.value_estimates -= self.alpha*(reward - self.averaged_reward)*pi
        self.value_estimates[last_action] = ht

    def reset(self):
        """Reset memory"""
        super(GradientAgent, self).reset()
        
        self.averaged_reward = 0


class BetaAgent(Agent):
    """Agent class with a bayesian approach policy"""
    def __init__(
        self, 
        policy, 
        num_arm, 
        num_trial, 
        b_use_thompson_sampling=True
    ):
        super(BetaAgent, self).__init__(policy=policy, num_arm=num_arm)

        # Parameters
        self.num_trial = num_trial
        self.b_use_thompson_sampling = b_use_thompson_sampling
        self.alphas = np.ones((self.num_arm,))
        self.betas = np.ones((self.num_arm,))

    def __str__(self):
        return "Beta Agent"

    def observe(self, reward):
        """Observe enrivonment"""
        last_action = self.action_history[-1]

        self.alphas[last_action] += reward
        self.betas[last_action] += self.num_trial - reward

        if self.b_use_thompson_sampling:
            self.value_estimates = np.random.beta(
                a=self.alphas, b=self.betas, size=(self.num_arm,)
            )
        else:
            self.value_estimates = self.alphas / (self.alphas + self.betas)

    def reset(self):
        """Reset memory"""
        super(BetaAgent, self).reset()

        self.alphas = np.ones((self.num_arm,))
        self.betas = np.ones((self.num_arm,))

