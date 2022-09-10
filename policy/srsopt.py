import numpy as np
import copy

class SRS_OPT(object):
    def __init__(self, K):
        self.epsilon = 1e-4
        self.K = K
        self.aleph = None

    def initialize(self):
        self.V = np.array([0.5] * self.K)
        cpV = copy.deepcopy(self.V)
        if self.V.max() > self.aleph: cpV -= cpV.max() - self.aleph + self.epsilon
        self.n = np.array([self.epsilon] * self.K)
        self.N = np.sum(self.n)
        self.Z = 1 / (np.sum(1 / (self.aleph - cpV)))
        self.rho = self.Z / (self.aleph - cpV)
        self.b = self.n / self.rho - self.N + self.epsilon
        self.SRS = (self.N + self.b.max()) * self.rho - self.n
        self.pi = self.SRS / np.sum(self.SRS)

    def select_arm(self):
        arm = np.random.choice(len(self.pi), p=self.pi)
        return arm

    def update(self, arm, reward):
        self.alpha = 1 / (1 + self.n[arm])
        self.V[arm] = (1 - self.alpha) * self.V[arm] + (reward * self.alpha)
        cpV = copy.deepcopy(self.V)
        if self.V.max() > self.aleph: cpV -= cpV.max() - self.aleph + self.epsilon
        self.n[arm] += 1
        self.N += 1
        self.Z = 1 / (np.sum(1 / (self.aleph - cpV)))
        self.rho = self.Z / (self.aleph - cpV)
        self.b = self.n / self.rho - self.N + self.epsilon
        self.SRS = (self.N + self.b.max()) * self.rho - self.n
        self.pi = self.SRS / np.sum(self.SRS)