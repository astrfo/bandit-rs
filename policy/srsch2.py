import numpy as np
import copy
import sys

class SRS_CH2(object):
    def __init__(self, K):
        self.epsilon = sys.float_info.epsilon
        self.K = K

    def initialize(self):
        self.aleph = 1.0
        self.V = np.array([0.5] * self.K)
        self.n = np.array([self.epsilon] * self.K)
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
        self.N += 1
        self.V[arm] += (1/(self.n[arm]+1)) * (reward - self.V[arm])

        # if np.amax(self.V) >= self.aleph:
        #     fix_aleph = np.amax(self.V) + self.epsilon
        #     diff = fix_aleph - self.V
        #     if np.amin(diff) < 0:
        #         diff -= np.amin(diff)
        #     self.Z = 1 / np.sum(1 / diff)
        #     self.rho = self.Z / diff
        #     self.b = self.n/self.rho - self.N + self.epsilon
        #     self.SRS = (self.N + np.amax(self.b)) * self.rho - self.n
        #     if np.amin(self.SRS) < 0: self.SRS -= np.amin(self.SRS)
        #     self.pi = self.SRS / np.sum(self.SRS)
        # else:
        #     self.Z = 1 / np.sum(1 / (self.aleph - self.V))
        #     self.rho = self.Z / (self.aleph - self.V)
        #     self.b = self.n/self.rho - self.N + self.epsilon
        #     self.SRS = (self.N + np.amax(self.b)) * self.rho - self.n
        #     if np.amin(self.SRS) < 0: self.SRS -= np.amin(self.SRS)
        #     self.pi = self.SRS / np.sum(self.SRS)
        # self.update_aleph()

        if np.amax(self.V) >= self.aleph:
            srs = np.zeros(self.K)
            is_satisfied = (self.V >= self.aleph)
            # alephとQ値が同値の場合に0除算が発生してしまうのをケア
            rs_value_plus_eps = (self.n/self.N) * (self.V - self.aleph) + self.epsilon

            for i, b in enumerate(is_satisfied):
                if b:
                    srs[i] = rs_value_plus_eps[i] / np.sum(rs_value_plus_eps[is_satisfied])
            self.SRS = srs
        else:
            self.Z = 1 / np.sum(1 / (self.aleph - self.V))
            self.rho = self.Z / (self.aleph - self.V)
            self.b = self.n/self.rho - self.N + self.epsilon
            self.SRS = (self.N + np.amax(self.b)) * self.rho - self.n
            if np.amin(self.SRS) < 0: self.SRS -= np.amin(self.SRS)
        self.pi = self.SRS / np.sum(self.SRS)
        self.update_aleph()
        
    def update_aleph(self):
        G = np.random.choice(np.where(self.V == self.V.max())[0])
        mu = np.exp(-self.n * self.D_KL(self.V, self.V[G]))
        aleph_list = self.V[G] * (1 - (self.V/self.V[G]) * mu) / (1 - mu)
        np.nan_to_num(aleph_list, copy=False)
        self.aleph = np.amax(aleph_list)
        
    def D_KL(self, p, q):
        return p*np.log(p/q) + (1-p)*np.log((1-p) / (1-q))