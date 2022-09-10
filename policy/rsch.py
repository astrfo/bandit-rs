import numpy as np

class RS_CH(object):
    def __init__(self, K):
        self.K = K

    def initialize(self):
        self.V = np.array([0.5] * self.K)
        self.n = np.array([1e-4] * self.K)
        self.N = np.sum(self.n)
        self.aleph = np.ones(self.K)
        self.RS = (self.n / self.N) * (self.V - self.aleph)

    def select_arm(self):
        G = np.random.choice(np.where(self.V == self.V.max())[0])
        RSG = (self.n[G] / self.N) * (self.V[G] - self.aleph)
        mu = np.exp(-self.n * self.D_KL(self.V, self.V[G]))
        mu[G] = 0.0
        self.aleph = self.V[G] * (1 - (self.V/self.V[G]) * mu) / (1 - mu)
        np.nan_to_num(self.aleph, copy=False, nan=0)
        exceed_RSG_index = np.where(RSG <= self.RS)[0]

        if len(exceed_RSG_index) == 1:
            arm = G
        else:
            if len(np.where(RSG < self.RS)[0]) == 1:
                arm = np.where(RSG < self.RS)[0][0]
            else:
                exceed_mu = mu[exceed_RSG_index]
                max_mu_index = np.where(exceed_mu.max() == exceed_mu)[0]
                if len(max_mu_index) == 1:
                    arm = exceed_RSG_index[max_mu_index[0]]
                else:
                    exceed_aleph = self.aleph[max_mu_index]
                    max_aleph_index = np.where(exceed_aleph.max() == exceed_aleph)[0]
                    if len(max_aleph_index) == 1:
                        arm = exceed_RSG_index[max_mu_index[max_aleph_index[0]]]
                    else:
                        arm = np.random.choice(exceed_RSG_index[max_mu_index[max_aleph_index]])
        return arm

    def D_KL(self, p, q):
        return p*np.log(p/q) + (1-p)*np.log((1-p) / (1-q))

    def update(self, arm, reward):
        self.alpha = 1 / (1 + self.n[arm])
        self.V[arm] = (1 - self.alpha) * self.V[arm] + (reward * self.alpha)
        self.n[arm] += 1
        self.N += 1
        self.RS = (self.n / self.N) * (self.V - self.aleph)