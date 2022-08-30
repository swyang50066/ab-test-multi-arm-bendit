from abc import ABC, abstractmethod

import numpy as np


class MultiArmedBanditWrapper(ABC):
    """Multi-Armed Bandit Machine Class"""
    def __init__(self, num_arm):
        # Parameters
        self.num_arm = num_arm
        
        # Value spaces
        self.action_values = np.zeros((self.num_arm,))
    
    def get_optimal_value(self):
        """"Return current optimal value"""
        return np.max(self.action_values)

    def get_optimal_action(self):
        """Return current optimal action"""
        return np.argmax(self.action_values)
    
    @abstractmethod
    def setup(self):
        """Set up bandit action values"""
        pass

    @abstractmethod
    def pull_arm(self, selection):
        """Return reward pulling the selected bandit arm"""
        pass


class DeterministicBandit(MultiArmedBanditWrapper):
    """Deterministic bandit"""
    def __init__(self, num_arm, reward_scale=1):
        super(DeterministicBandit, self).__init__(num_arm=num_arm)

        # Parameter
        self.reward_scale

        # Set up bandit state
        self.setup()

    def setup(self):
        """Set up bandit action values"""
        pass

    def pull_arm(self, selection):
        """Return reward pulling the selected bandit arm"""
        return self.reward_scale, True


class GaussianBandit(MultiArmedBanditWrapper):
    """Bandit with Gaussian reward distribution"""
    def __init__(self, num_arm, mean_reward=0, stddev_reward=1):
        super(GaussianBandit, self).__init__(num_arm=num_arm)

        # Parameters
        self.mean_reward = mean_reward
        self.stddev_reward = stddev_reward

        # Set up bandit state
        self.setup()

    def setup(self):
        """Set up bandit action values"""
        self.action_values = np.random.normal(
            loc=self.mean_reward,
            scale=self.stddev_reward,
            size=self.num_arm
        )

    def pull_arm(self, selection):
        """Return reward pulling the selected bandit arm"""
        return (
            np.random.normal(
                loc=self.action_values[selection], 
                scale=self.stddev_reward
            ), 
            selection == self.get_optimal_action
        )


class BinomialBandit(MultiArmedBanditWrapper):
    """Bandit with Binomial reward distribution"""
    def __init__(self, num_arm, num_trial, num_sample, probs=None):
        super(BinomialBandit, self).__init__(num_arm=num_arm)

        # Paramters
        self.num_trial = num_trial
        self.num_sample = num_sample
        self.probs = probs 

        # Set up bandit state
        self.setup()

    def setup(self):
        """Set up bandit action values"""
        self.action_values = (
            np.random.uniform(size=self.num_arm)
            if self.probs is None else self.probs
        )
        
        if self.num_sample is not None:
            self._samples = np.random.binomial(
                n=self.num_trial*np.ones(self.num_arm, dtype=np.int),
                p=self.action_values,
                size=(self.num_sample, self.num_arm)
            )
        
        self._cursor = -1

    def pull_arm(self, selection):
        """Return reward pulling the selected bandit arm"""
        return self.sample[selection], selection == self.get_optimal_action

    @property
    def sample(self):
        if self._samples is None:
            return np.random.binomial(
                n=self.num_trial*np.ones(self.num_arm, dtype=np.int),
                p=self.action_values,
                size=(1, self.num_arm)
            )
        else:
            self._cursor += 1
            return self._samples[self._cursor]
 
