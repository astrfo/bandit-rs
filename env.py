import numpy as np

class Environment(object):
    def __init__(self, K, seed):
        self.K = K
        np.random.seed(seed)
        self.prob = np.random.rand(self.K)

    def play(self, arm):
        if self.prob[arm] > np.random.rand():
            return 1
        else:
            return 0