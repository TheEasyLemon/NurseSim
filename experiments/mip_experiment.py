'''
Experiment with the MIP solver in ProblemSolver/mip.py.
'''
import sys

import matplotlib.pyplot as plt
import numpy as np

from ProblemSolver.mip import MIP
from ProblemSolver.hm1 import HM1
from ProblemSolver.ProblemSolver import ProblemSolver
from utils.random_pi import generate_random_pi


def test_batch(m, Nj, rounds=1000):
    correct = 0
    rev_total = 0
    rev_captured = 0

    for i in range(rounds):
        pi = generate_random_pi(m, 1, Nj=Nj, constant_revenue_per_shift=True, round=2)
        mip_soln = HM1(pi, k=1)
        opt_soln = ProblemSolver(pi).optimalPolicy()
        mip_rev = pi.expectedRevenueCol(0, mip_soln[:, 0]) 
        opt_rev = pi.expectedRevenueCol(0, opt_soln[:, 0])

        if abs(mip_rev - opt_rev) < sys.float_info.epsilon:
            correct += 1

        rev_total += opt_rev
        rev_captured += mip_rev
        
        # print(mip_soln)
        # print(opt_soln)
        # print(pi.expectedRevenueCol(0, mip_soln[:, 0]))
        # print(pi.expectedRevenueCol(0, opt_soln[:, 0]))

    rev_dev = 1 - rev_captured / rev_total
    accuracy = correct / rounds
    print('Revenue deviation:', rev_dev)
    print('Found optimal solution at this percentage:', accuracy)

    return rev_dev, accuracy

if __name__ == '__main__':
    Nj = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    rev_devs = np.zeros(Nj.size)
    accs = np.zeros(Nj.size)

    for i, nj in enumerate(Nj):
        print('Starting for nj=', nj)
        rev_dev, acc = test_batch(8, int(nj), rounds=10000)
        rev_devs[i] = rev_dev
        accs[i] = acc

    plt.plot(Nj, rev_devs)
    plt.xlabel('$N_j$')
    plt.ylabel('Revenue Deviation from Optimal')
    plt.title('Revenue Deviation as a function of $N_j$ for $m=8$')
    plt.show()

    plt.plot(Nj, accs)
    plt.xlabel('$N_j$')
    plt.ylabel('Accuracy')
    plt.title('Accuracy as a function of $N_j$ for $m=8$')
    plt.show()