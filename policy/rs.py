import numpy as np

class RS(object):
    def __init__(self, K):
        self.K = K
        self.aleph = None

    def initialize(self):
        self.V = np.array([0.5] * self.K)
        self.n = np.array([1e-4] * self.K)
        self.N = np.sum(self.n)
        self.RS = (self.n / self.N) * (self.V - self.aleph)

    def select_arm(self):
        arm = np.random.choice(np.where(self.RS == self.RS.max())[0])
        return arm

    def update(self, arm, reward):
        self.alpha = 1 / (1 + self.n[arm])
        self.V[arm] = (1 - self.alpha) * self.V[arm] + (reward * self.alpha)
        self.n[arm] += 1
        self.N += 1
        self.RS = (self.n / self.N) * (self.V - self.aleph)