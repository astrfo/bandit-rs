import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot(args):
    csv_files = glob.glob('./log/' + args[2] + '/*.csv')
    for file in csv_files:
        name = os.path.splitext(os.path.basename(file))[0]
        data = pd.read_csv(file)
        plt.plot(data, label=name)
    plt.legend()
    plt.title(args[1])
    plt.xlabel('step')
    plt.ylabel('regret')
    plt.savefig('./log/' + args[2] + '/csv_plot.png')
    plt.show()

if __name__ == '__main__':
    args = sys.argv
    if len(args) <= 1:
        print('wrong number of arguments')
        sys.exit()
    plot(args)