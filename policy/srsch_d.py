import numpy as np
import copy
import sys
from decimal import Decimal, getcontext, FloatOperation, InvalidOperation, DivisionByZero, Overflow

getcontext().traps[FloatOperation] = False
getcontext().traps[InvalidOperation] = False 
getcontext().traps[DivisionByZero] = False
getcontext().traps[Overflow] = False

class SRS_CHd(object):
    def __init__(self, K):
        self.epsilon = Decimal('1e-6')
        self.K = K

    def initialize(self):
        self.aleph = np.array([Decimal('1.0')] * self.K)
        self.V = np.array([Decimal('0.5')] * self.K)
        self.n = np.array([Decimal('0.0')] * self.K)
        self.N = np.array([Decimal('0.0')] * self.K)
        self.Z = np.array([Decimal('0.0')] * self.K)
        self.rho = np.array([[Decimal('0.0')] * 2 for _ in range(self.K)])
        self.b = np.array([[Decimal('0.0')] * 2 for _ in range(self.K)])
        self.SRS = np.array([[Decimal('0.0')] * 2 for _ in range(self.K)])
        self.pi = np.array([Decimal('0.0')] * self.K)
        self.pipi = np.array([[Decimal('0.0')] * 2 for _ in range(self.K)])
        self.mu = np.array([Decimal('0.0')] * self.K)

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

    # def D_KL(self, p, q):
    #     return p*np.log(p/q) + (1-p)*np.log((1-p) / (1-q))

    def D_KL(self, p, q):
        return p*Decimal(p/q).ln() + (1-p)*Decimal((1-p) / (1-q)).ln()
        
    def update(self, arm, reward):
        self.n[arm] += 1
        self.V[arm] += (1 / (1 + self.n[arm])) * (reward - self.V[arm])
        G = np.random.choice(np.where(self.V == self.V.max())[0])
        self.update_aleph(G)

        for i in range(self.K):
            # if i != G:
            #     v2 = np.array([self.V[i], self.V[G]])
            #     n2 = np.array([self.n[i], self.n[G]])
            #     n2_total = self.n[i] + self.n[G]
            #     if v2[1] >= self.aleph[i]:
            #         fix_aleph = v2[1] + self.epsilon
            #         diff = fix_aleph - v2
            #         if np.amin(diff) < 0:
            #             diff -= np.amin(diff)
            #         z2 = 1 / np.sum(1 / diff)
            #         rho2 = z2 / diff
            #         b2 = n2/rho2 - n2_total + self.epsilon
            #         srs2 = (n2_total + np.amax(b2)) * rho2 - n2
            #         if np.amin(srs2) < 0: srs2 -= np.amin(srs2)
            #     else:
            #         diff = self.aleph[i] - v2
            #         z2 = 1 / np.sum(1 / diff)
            #         rho2 = z2 / diff
            #         b2 = n2/rho2 - n2_total + self.epsilon
            #         srs2 = (n2_total + np.amax(b2)) * rho2 - n2
            #         if np.amin(srs2) < 0: srs2 -= np.amin(srs2)
            #     # if np.isnan(srs2[0]) or np.isinf(srs2[0]): srs2[0] = self.epsilon
            #     # if np.isnan(srs2[1]) or np.isinf(srs2[1]): srs2[1] = self.epsilon
            #     if srs2[0].is_nan() or srs2[0].is_infinite(): srs2[0] = self.epsilon
            #     if srs2[1].is_nan() or srs2[1].is_infinite(): srs2[1] = self.epsilon
            #     self.pipi[i] = srs2 / np.sum(srs2)
            #     if self.pipi[i][0] <= 0: self.pipi[i][0] = self.epsilon
            #     if self.pipi[i][1] <= 0: self.pipi[i][1] = self.epsilon
            # else:
            #     self.pipi[G] = Decimal('0.5')
            
            if i != G:
                v2 = np.array([self.V[i], self.V[G]])
                n2 = np.array([self.n[i], self.n[G]])
                n2_total = self.n[i] + self.n[G]
                if v2[1] >= self.aleph[i]:
                    srs2 = np.array([Decimal('0.0')] * 2)
                    is_satisfied = (v2 >= self.aleph[i])
                    # alephとQ値が同値の場合に0除算が発生してしまうのをケア
                    rs_value_plus_eps = (n2/n2_total) * (v2 - self.aleph[i]) + self.epsilon

                    for i, b in enumerate(is_satisfied):
                        if b:
                            sum_rs_value_plus_eps = Decimal('0.0')
                            for j in rs_value_plus_eps[is_satisfied]:
                                sum_rs_value_plus_eps += j
                            srs2[i] = rs_value_plus_eps[i] / sum_rs_value_plus_eps
                else:
                    diff = self.aleph[i] - v2
                    z2 = 1 / np.sum(1 / diff)
                    rho2 = z2 / diff
                    b2 = n2/rho2 - n2_total + self.epsilon
                    srs2 = (n2_total + np.amax(b2)) * rho2 - n2
                    if np.amin(srs2) < 0: srs2 -= np.amin(srs2)
                # if np.isnan(srs2[0]) or np.isinf(srs2[0]): srs2[0] = self.epsilon
                # if np.isnan(srs2[1]) or np.isinf(srs2[1]): srs2[1] = self.epsilon
                if srs2[0].is_nan() or srs2[0].is_infinite(): srs2[0] = self.epsilon
                if srs2[1].is_nan() or srs2[1].is_infinite(): srs2[1] = self.epsilon
                self.pipi[i] = srs2 / np.sum(srs2)
                if self.pipi[i][0] <= 0: self.pipi[i][0] = self.epsilon
                if self.pipi[i][1] <= 0: self.pipi[i][1] = self.epsilon
            else:
                self.pipi[G] = Decimal('0.5')
        
        # sum_pi = 0.0
        sum_pi = Decimal('0.0')
        for i in range(self.K):
            sum_pi += self.pipi[i][0] / self.pipi[i][1]

        for i in range(self.K):
            self.pi[i] = (self.pipi[i][0]/self.pipi[i][1]) / sum_pi
    
    def update_aleph(self, G):
        for i in range(self.K):
            if i != G:
                self.mu[i] = np.exp(-self.n[i] * self.D_KL(self.V[i], self.V[G]))
                self.aleph[i] = self.V[G] * (1 - (self.V[i]/self.V[G]) * self.mu[i]) / (1 - self.mu[i])