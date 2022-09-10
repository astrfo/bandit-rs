import numpy as np

class UCB1_tuned(object):
    def __init__(self, K):
        self.K = K

    def initialize(self):
        self.n = np.zeros(self.K)
        self.V = np.array([0.5] * self.K)
        self.N = 0.0
        self.avg = np.zeros(self.K)
        self.var = np.zeros(self.K)
        self.reward_squares = np.zeros(self.K)
        self.ucb_v = np.zeros(self.K)
    
    def select_arm(self):
        for arm in range(self.K):
            if self.n[arm] == 0.0: return arm
        self.ucb_v = self.var + np.sqrt((2 * np.log(self.N)) / self.n)
        bonus = np.sqrt((np.log(self.N)) / self.n * np.array([x if x < 0.25 else 0.25 for x in self.ucb_v]))
        ucb = self.V + bonus
        return np.random.choice(np.where(ucb == ucb.max())[0])

    def update(self, arm, reward):
        self.alpha = 1 / (1 + self.n[arm])
        self.V[arm] = (1 - self.alpha) * self.V[arm] + (reward * self.alpha)
        self.n[arm] += 1
        self.N += 1
        self.reward_squares[arm] += reward**2
        self.avg[arm] += (reward - self.avg[arm]) / self.n[arm]
        self.var[arm] = (self.reward_squares[arm] / self.n[arm]) - self.avg[arm]**2
