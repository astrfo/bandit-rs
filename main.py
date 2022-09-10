from simulator import Simulator

def main():
    trial = 1000
    step = 1000
    K = 2
    sim = Simulator(trial, step, K)
    sim.run()

if __name__ == '__main__':
    print('started run')
    main()
    print('finished run')