import numpy as np
import copy
import sys

class SRS_CH(object):
    def __init__(self, K):
        self.epsilon = sys.float_info.epsilon
        self.K = K

    def initialize(self):
        self.mu = np.zeros(self.K)
        self.aleph = np.ones(self.K)
        self.Z = np.zeros(self.K)
        self.rho = np.zeros((self.K, 2))
        self.b = np.zeros((self.K, 2))
        self.SRS = np.zeros((self.K, 2))
        self.pipi = np.zeros((self.K, 2))
        self.pi = np.zeros(self.K)
        self.V = np.array([0.5] * self.K)
        self.n = np.array([self.epsilon] * self.K)
        self.N = np.zeros(self.K)

    def select_arm(self):
        G = np.random.choice(np.where(self.V == self.V.max())[0])
        cpV = copy.deepcopy(self.V)
        if (self.V.max() > self.aleph).all(): cpV -= cpV.max() - self.aleph + self.epsilon
        for i in range(self.K):
            if i != G:
                self.mu[i] = np.exp(-self.n[i] * self.D_KL(cpV[i], cpV[G]))
                self.aleph[i] = cpV[G] * (1 - (cpV[i]/cpV[G]) * self.mu[i]) / (1 - self.mu[i])

                self.Z[i] = 1 / ((self.aleph[i] - cpV[i]) + (self.aleph[i] - cpV[G]))

                self.rho[i][0] = self.Z[i] / (self.aleph[i] - cpV[i])
                self.rho[i][1] = self.Z[i] / (self.aleph[i] - cpV[G])

                self.N[i] = self.n[i] + self.n[G]

                self.b[i][0] = self.n[i] / self.rho[i][0] - self.N[i] + self.epsilon
                self.b[i][1] = self.n[G] / self.rho[i][1] - self.N[i] + self.epsilon

                self.SRS[i][0] = (self.N[i] + self.b[i].max()) * self.rho[i][0] - self.n[i]
                self.SRS[i][1] = (self.N[i] + self.b[i].max()) * self.rho[i][1] - self.n[G]
                self.SRS[i] = np.nan_to_num(self.SRS[i], nan=self.epsilon)

                self.pipi[i][0] = self.SRS[i][0] / (self.SRS[i][0] + self.SRS[i][1])
                self.pipi[i][1] = self.SRS[i][1] / (self.SRS[i][0] + self.SRS[i][1])
                if (self.pipi[i][0] <= 0.0): self.pipi[i][0] = self.epsilon
                if (self.pipi[i][1] <= 0.0): self.pipi[i][1] = self.epsilon
            else:
                self.pipi[G] = 0.5
        sum_pi = 0.0
        for i in range(self.K):
            sum_pi += self.pipi[i][0] / self.pipi[i][1]

        for i in range(self.K):
            self.pi[i] = (self.pipi[i][0]/self.pipi[i][1]) / sum_pi
        
        arm = np.random.choice(len(self.pi), p=self.pi)
        return arm

    def D_KL(self, p, q):
        return p*np.log(p/q) + (1-p)*np.log((1-p) / (1-q))
        
    def update(self, arm, reward):
        self.alpha = 1 / (1 + self.n[arm])
        self.V[arm] = (1 - self.alpha) * self.V[arm] + (reward * self.alpha)
        self.n[arm] += 1