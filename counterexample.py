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

def first_test():
    m = n = 5
    pi = generate_random_pi(m, n)
    pi_dynamic = pi.copy()
    pi_dynamic.N = np.ones(n, dtype=np.int64)
    ps = ProblemSolver(pi_dynamic)
    ps2 = ProblemSolver(pi)
    hh1_policy = HH1(pi)
    dynamic_policy = ps.dynamicPolicy()
    best_policy = ps2.optimalPolicy()
    print(pi.expectedRevenue(dynamic_policy))
    print(dynamic_policy)
    print(pi.N)
    print(pi.expectedRevenue(hh1_policy))
    print(hh1_policy)
    print(best_policy)
    print(pi.expectedRevenue(best_policy))

def compare_hh1_optimal(m: int, n: int) -> Tuple[np.ndarray, float]:
    pi = generate_random_pi(m, n)
    ps = ProblemSolver(pi)
    hh1_policy = HH1(pi)
    optimal_policy = ps.optimalPolicy()
    shift_difference = np.logical_xor(hh1_policy, optimal_policy)
    revenue_difference = pi.expectedRevenue(optimal_policy) - pi.expectedRevenue(hh1_policy)
    return shift_difference, revenue_difference

def analyze_hh1_optimal(m: int, n: int):
    N = 1000
    total_shift = np.zeros((m, n))
    total_rev = 0

    print(f'Running {N} replications...')

    for _ in range(N):
        shift_diff, rev_diff = compare_hh1_optimal(m, n)
        total_shift += shift_diff
        total_rev += rev_diff

    print('Probability that this cell differs from optimal policy:')
    print(total_shift / N)

    print(f'Average revenue deviation: {total_rev / N}')

if __name__ == '__main__':
    m = n = 5
    analyze_hh1_optimal(m, n)
