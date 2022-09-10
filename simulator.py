import os
from datetime import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
from env import Environment
from policy.rs import RS
from policy.rsopt import RS_OPT
from policy.rsch import RS_CH
from policy.srs import SRS
from policy.srsopt import SRS_OPT
from policy.srsch import SRS_CH
from policy.ts import TS
from policy.ucb1 import UCB1
from policy.ucb1_tuned import UCB1_tuned
import warnings
warnings.simplefilter('ignore', category=RuntimeWarning)

class Simulator(object):
    def __init__(self, trial, step, K):
        self.policy = {RS(K): 'RS', RS_OPT(K): 'RS_OPT', RS_CH(K): 'RS_CH', SRS(K): 'SRS', SRS_OPT(K): 'SRS_OPT', SRS_CH(K): 'SRS_CH', TS(K): 'TS', UCB1(K): 'UCB1', UCB1_tuned(K): 'UCB1_tuned'}
        self.policy_plot_name = ['RS($ℵ=p_{max}$)', 'RS-OPT', 'RS-CH', 'SRS($ℵ=p_{max}$)', 'SRS-OPT', 'SRS-CH', 'TS', 'UCB1', 'UCB1-tuned']
        self.trial = trial
        self.step = step
        self.K = K
        self.regret = np.zeros(self.step)
        self.fig = plt.plot()

    def run(self):
        time_now = datetime.now()
        results_dir = f'log/{time_now:%Y%m%d%H%M}/'
        os.makedirs(results_dir, exist_ok=True)
        f = open(results_dir + 'log.txt', mode='w', encoding='utf-8')
        for policy, name in self.policy.items():
            start = time.time()
            f.write(f'sim: {self.trial}, step: {self.step}, K: {self.K}\n')
            for t in range(self.trial):
                self.env = Environment(self.K)
                self.prob = self.env.prob
                self.setting(name, policy)
                policy.initialize()
                self.regretV = 0.0
                for s in range(self.step):
                    arm = policy.select_arm()
                    reward = self.env.play(arm)
                    policy.update(arm, reward)
                    self.calc_regret(t, s, arm)
            end = time.time() - start
            f.write(f'{name}:\t {end}[sec]\n')
            print(f'{name}:\t {end}[sec]')
            self.print_regret()
            plt.savefig(results_dir + 'results.png')
            np.savetxt(results_dir + name + '.csv', self.regret, delimiter=",")
        f.close()
        plt.show()

    def setting(self, name, policy):
        if name == 'SRS' or name == 'RS':
            policy.aleph = self.prob.max()
        if name == 'RS_OPT' or name == 'SRS_OPT': 
            policy.aleph = sum(sorted(self.prob, reverse=True)[:2]) / 2

    def calc_regret(self, t, s, arm):
        self.regretV += (self.prob.max() - self.prob[arm])
        self.regret[s] += (self.regretV - self.regret[s]) / (t+1)

    def print_regret(self):
        plt.plot(np.arange(self.step), self.regret)
        plt.title(f'sim: {self.trial}, step: {self.step}, K: {self.K}')
        plt.xlabel('steps')
        plt.ylabel('regret')
        plt.legend(labels=self.policy_plot_name, loc='lower right')