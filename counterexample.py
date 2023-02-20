'''
Finds counterexamples to the HH1 algorithm.

Dawson Ren
2/6/23
'''
from typing import Tuple

import numpy as np

from ProblemSolver.hh1 import HH1
from ProblemSolver.ProblemSolver import ProblemSolver
from utils.random_pi import generate_random_pi

def first_test(m: int, n: int):
    pi = generate_random_pi(m, n)
    ps = ProblemSolver(pi)
    dynamic_policy, dynamic_rev = ps.dynamicColumn2(0)
    optimal_policy = ps.optimalColumn(0)
    print(pi.expectedRevenueCol(0, dynamic_policy))
    print('Dynamic rev: ', dynamic_rev)
    print(dynamic_policy)
    print(pi.N[0])
    print(pi.expectedRevenueCol(0, optimal_policy))
    print(optimal_policy)

def compare_hh1_optimal(m: int, n: int) -> Tuple[np.ndarray, float]:
    pi = generate_random_pi(m, n)
    ps = ProblemSolver(pi)
    hh1_policy = HH1(pi)
    optimal_policy = ps.optimalPolicy()
    shift_difference = np.logical_xor(hh1_policy, optimal_policy)
    rev_opt = pi.expectedRevenue(optimal_policy) 
    revenue_difference = rev_opt - pi.expectedRevenue(hh1_policy)
    return shift_difference, revenue_difference, rev_opt

def analyze_hh1_optimal(m: int, n: int):
    N = 10
    total_shift = np.zeros((m, n))
    total_rev = 0
    total_avg_rev = 0

    print(f'Running {N} replications...')

    for _ in range(N):
        shift_diff, rev_diff, rev_opt = compare_hh1_optimal(m, n)
        total_shift += shift_diff
        total_rev += rev_diff
        total_avg_rev += rev_opt

    print('Probability that this cell differs from optimal policy:')
    print(total_shift / N)

    print(f'Average revenue deviation: {total_rev / N}')
    print(f'Average revenue: {total_avg_rev / N}')
    print(f'Average percent revenue deviation: {total_rev / total_avg_rev}')

if __name__ == '__main__':
    m = n = 3
    first_test(m, n)
    # analyze_hh1_optimal(m, n)
