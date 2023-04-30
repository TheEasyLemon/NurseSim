'''
Find the optimality gap between the binary/relaxed MIP solvers.
'''
import numpy as np
import sys

from utils.random_pi import generate_random_pi
from ProblemSolver.mip import MIP
from ProblemSolver.ProblemSolver import ProblemSolver

FILE_OUT = 'experiments/mip_optimality_gap.txt'
M = 5
Nj = 1

def write_revenue(rounds=1000):
    '''
    Write the revenue for each round as a column.
    The first row is the MIP binary Y version.
    The second row is the MIP relaxed version.
    '''
    revenues = np.zeros((2, rounds))

    for i in range(rounds):
        if i % 100 == 0:
            print(i)
        pi = generate_random_pi(M, 1, Nj=Nj, constant_revenue_per_shift=True, round=2)
        mip_soln = MIP(pi)
        mip_soln_relaxed = MIP(pi, binary_y=False)
        revenues[0, i] = pi.expectedRevenueCol(0, mip_soln[:, 0]) 
        revenues[1, i] = pi.expectedRevenueCol(0, mip_soln_relaxed[:, 0])


    np.savetxt(FILE_OUT, revenues)

def find_optimality_gap():
    revenues = np.loadtxt(FILE_OUT, delimiter=' ')
    # eliminate scenarios where both revenues were 0
    revenues = revenues[:, np.max(revenues, axis=0) > 0]
    binary_rev = revenues[0, :] 
    relaxed_rev = revenues[1, :] 
    best_rev = np.max(revenues, axis=0)
    print(f'All results for m={M}, Nj={Nj}')
    print('Average relative optimality gap:', np.mean((binary_rev - relaxed_rev) / best_rev))
    print('Standard deviation of relative optimality gap:', np.std((binary_rev - relaxed_rev) / best_rev))
    print('Proportion of the time binary was better:', np.mean(binary_rev == best_rev))

if __name__ == '__main__':
    # write_revenue()
    find_optimality_gap()
