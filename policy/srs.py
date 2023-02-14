import numpy as np
import copy
import sys

class SRS(object):
    def __init__(self, K):
        self.epsilon = sys.float_info.epsilon
        self.K = K
        self.aleph = None

    def initialize(self):
        self.V = np.array([0.5] * self.K)
        self.n = np.zeros(self.K)
        self.N = 0
        self.Z = 0
        self.rho = np.zeros(self.K)
        self.b = np.zeros(self.K)
        self.SRS = np.zeros(self.K)
        self.pi = np.array([1/self.K] * self.K)

    def select_arm(self):
        current_prob = np.random.rand()
        top = self.K
        bottom = -1
        while (top - bottom > 1):
            mid = int(bottom + (top - bottom)/2)
            if current_prob < np.sum(self.pi[0:mid]): top = mid
            else: bottom = mid
        if mid == bottom: arm = mid
        else: arm = mid-1
        return arm

    def update(self, arm, reward):
        self.n[arm] += 1
        self.V[arm] += (1 / (1 + self.n[arm])) * (reward - self.V[arm])
        self.N += 1
        if np.amax(self.V) >= self.aleph:
            fix_aleph = np.amax(self.V) + self.epsilon
            diff = fix_aleph - self.V
            if np.amin(diff) < 0:
                diff -= np.amin(diff)
            self.Z = 1 / np.sum(1 / diff)
            self.rho = self.Z / diff
            self.b = self.n/self.rho - self.N + self.epsilon
            self.SRS = (self.N + np.amax(self.b)) * self.rho - self.n
            if np.amin(self.SRS) < 0: self.SRS -= np.amin(self.SRS)
            self.pi = self.SRS / np.sum(self.SRS)
        else:
            self.Z = 1 / np.sum(1 / (self.aleph - self.V))
            self.rho = self.Z / (self.aleph - self.V)
            self.b = self.n/self.rho - self.N + self.epsilon
            self.SRS = (self.N + np.amax(self.b)) * self.rho - self.n
            if np.amin(self.SRS) < 0: self.SRS -= np.amin(self.SRS)
            self.pi = self.SRS / np.sum(self.SRS)