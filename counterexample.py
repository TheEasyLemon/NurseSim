'''
Finds counterexamples to the HH1 algorithm.

Dawson Ren
2/6/23
'''
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from ProblemSolver.hh1 import HH1, HH2
from ProblemSolver.hm1 import HM1
from ProblemSolver.ProblemSolver import ProblemSolver
from utils.random_pi import generate_random_pi

def compare_hh_optimal(m: int, n: int) -> Tuple[np.ndarray, float]:
    pi = generate_random_pi(m, n)
    ps = ProblemSolver(pi)

    hh1_policy = HH1(pi)
    hh2_policy = HH2(pi)
    hm1_policy = HM1(pi, 1)
    optimal_policy = ps.optimalPolicy()

    shift_difference_hh1 = np.logical_xor(hh1_policy, optimal_policy)
    shift_difference_hh2 = np.logical_xor(hh2_policy, optimal_policy)
    shift_difference_hm1 = np.logical_xor(hm1_policy, optimal_policy)

    rev_opt = pi.expectedRevenue(optimal_policy)

    revenue_hh1 = pi.expectedRevenue(hh1_policy)
    revenue_hh2 = pi.expectedRevenue(hh2_policy)
    revenue_hm1 = pi.expectedRevenue(hm1_policy)

    return (
        shift_difference_hh1, revenue_hh1,
        shift_difference_hh2, revenue_hh2,
        shift_difference_hm1, revenue_hm1,
        rev_opt
    )

def analyze_hh1_optimal(m: int, n: int, verbose=False):
    N = 500
    total_shift_hh1 = np.zeros((m, n))
    total_rev_hh1 = 0
    total_shift_hh2 = np.zeros((m, n))
    total_rev_hh2 = 0
    total_shift_hm1 = np.zeros((m, n))
    total_rev_hm1 = 0
    total_rev = 0

    print(f'Running {N} replications...')

    for _ in range(N):
        shift_diff_hh1, rev_hh1, shift_diff_hh2, rev_hh2, shift_diff_hm1, rev_hm1, rev_opt = compare_hh_optimal(m, n)
        total_shift_hh1 += shift_diff_hh1
        total_rev_hh1 += rev_hh1
        total_shift_hh2 += shift_diff_hh2
        total_rev_hh2 += rev_hh2
        total_shift_hm1 += shift_diff_hm1
        total_rev_hm1 += rev_hm1
        total_rev += rev_opt
    
    prop_rev_deviation_hh1 = (total_rev - total_rev_hh1) / total_rev
    prop_rev_deviation_hh2 = (total_rev - total_rev_hh2) / total_rev
    prop_rev_deviation_hm1 = (total_rev - total_rev_hm1) / total_rev

    if verbose:
        print('Probability that this cell differs from optimal policy:')
        print(f'HH1:\n{total_shift_hh1 / N}')
        print(f'HH2:\n{total_shift_hh2 / N}')
        print(f'HM1:\n{total_shift_hm1 / N}')

        print(f'Average revenue deviation for hh1: {(total_rev - total_rev_hh1) / N}')
        print(f'Average revenue deviation for hh2: {(total_rev - total_rev_hh2) / N}')
        print(f'Average revenue deviation for hm1: {(total_rev - total_rev_hm1) / N}')

        print(f'Average revenue: {total_rev / N}')

        print(f'Average proportion of revenue deviation for hh1: {prop_rev_deviation_hh1}')
        print(f'Average proportion of revenue deviation for hh2: {prop_rev_deviation_hh2}')
        print(f'Average proportion of revenue deviation for hm1: {prop_rev_deviation_hm1}')

    return prop_rev_deviation_hh1, prop_rev_deviation_hh2, prop_rev_deviation_hm1

def run_comparison_graph():
    size = np.arange(4, 5)
    size = size.reshape(size.size, 1)
    prop_dev = np.zeros((size.size, 3))
    for i, s in enumerate(size):
        hh1, hh2, hm1 = analyze_hh1_optimal(int(s), 1)
        prop_dev[i, 0] = hh1
        prop_dev[i, 1] = hh2
        prop_dev[i, 2] = hm1

    np.savetxt('hh1_hh2_hm1_2.csv', np.hstack((size, prop_dev)))

def display_comparison_graph():
    data = np.loadtxt('hh1_hh2_hm1_2.csv')
    size = data[:, 0]
    prop_dev = data[:, [1, 2, 3]]
    
    plt.title('Proportion Deviation from Optimal vs Problem Size ($n$)')
    plt.ylabel('Proportion Deviation from Optimal')
    plt.xlabel('Problem Size ($n$)')
    plt.plot(size, prop_dev[:, 0], label='HH1')
    plt.plot(size, prop_dev[:, 1], label='HH1+')
    plt.plot(size, prop_dev[:, 2], label='HM1')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run_comparison_graph()
    # display_comparison_graph()
    

