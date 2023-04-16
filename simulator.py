import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
from env import Environment
from policy.rs import RS
from policy.rsopt import RS_OPT
from policy.rsch import RS_CH
from policy.srs import SRS
from policy.srsopt import SRS_OPT
from policy.srsch import SRS_CH
from policy.srsch2 import SRS_CH2
from policy.srsch_a import SRS_CHa
from policy.srsch_d import SRS_CHd
from policy.ts import TS
from policy.ucb1 import UCB1
from policy.ucb1_tuned import UCB1_tuned
import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)

class Simulator(object):
    def __init__(self, trial, step, K):
        # self.policy = {RS(K): 'RS', RS_OPT(K): 'RS_OPT', RS_CH(K): 'RS_CH', SRS(K): 'SRS', SRS_OPT(K): 'SRS_OPT', SRS_CH(K): 'SRS_CH', TS(K): 'TS', UCB1(K): 'UCB1', UCB1_tuned(K): 'UCB1_tuned'}
        self.policy = {SRS_CHd(K): 'SRS_CHd'}
        self.trial = trial
        self.step = step
        self.K = K
        self.regret = np.zeros(self.step)
        self.make_folder()

    def run(self):
        for policy, name in self.policy.items():
            for t in tqdm(range(self.trial)):
                self.env = Environment(self.K, t)
                self.prob = self.env.prob
                self.setting(name, policy)
                policy.initialize()
                self.regretV = 0.0
                for s in range(self.step):
                    arm = policy.select_arm()
                    reward = self.env.play(arm)
                    policy.update(arm, reward)
                    self.calc_regret(t, s, arm)
            self.save_csv(name)

    def setting(self, name, policy):
        if name == 'SRS' or name == 'RS':
            policy.aleph = self.prob.max()
        if name == 'RS_OPT' or name == 'SRS_OPT': 
            policy.aleph = sum(sorted(self.prob, reverse=True)[:2]) / 2

    def calc_regret(self, t, s, arm):
        self.regretV += (self.prob.max() - self.prob[arm])
        self.regret[s] += (self.regretV - self.regret[s]) / (t+1)

    def make_folder(self):
        time_now = datetime.now()
        self.results_dir = f'log/{time_now:%Y%m%d%H%M}/'
        os.makedirs(self.results_dir, exist_ok=True)

    def save_csv(self, name):
        f = open(self.results_dir + 'log.txt', mode='w', encoding='utf-8')
        f.write(f'sim: {self.trial}, step: {self.step}, K: {self.K}\n')
        np.savetxt(self.results_dir + name + '.csv', self.regret, delimiter=",")
        f.close()