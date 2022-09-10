import numpy as np

class TS(object):
    def __init__(self, K):
        self.K = K

    def initialize(self):
        self.S = np.ones(self.K)
        self.F = np.ones(self.K)

    def select_arm(self):
        theta = np.array([np.random.beta(self.S[i], self.F[i]) for i in range(self.K)])
        arm = np.random.choice(np.where(theta == theta.max())[0])
        return arm

    def update(self, arm, reward):
        if reward == 1:
            self.S[arm] += 1
        else:
            self.F[arm] += 1