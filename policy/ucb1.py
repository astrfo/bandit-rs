import numpy as np

class UCB1(object):
    def __init__(self, K):
        self.K = K

    def initialize(self):
        self.n = np.zeros(self.K)
        self.V = np.array([0.5] * self.K)
        self.N = 0.0
    
    def select_arm(self):
        for arm in range(self.K):
            if self.n[arm] == 0.0: return arm
        bonus = np.sqrt((2 * np.log(self.N)) / self.n)
        ucb = self.V + bonus
        return np.random.choice(np.where(ucb == ucb.max())[0])

    def update(self, arm, reward):
        self.alpha = 1 / (1 + self.n[arm])
        self.V[arm] = (1 - self.alpha) * self.V[arm] + (reward * self.alpha)
        self.n[arm] += 1
        self.N += 1