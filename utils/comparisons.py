'''
Compare problem solvers/heuristics.
'''
from typing import Tuple
import traceback

import numpy as np
import matplotlib.pyplot as plt

from ProblemSolver.id import IterativeDeepening
from ProblemSolver.ProblemSolver import ProblemSolver
from utils.random_pi import generate_random_pi

def compare_to_optimal(m: int, n: int, rounds: int, heuristics=None, use_iterative_deepening=False) -> Tuple[np.ndarray, float]:
    '''
    Inputs:
    - m: number of nurses
    - n: number of shifts
    - rounds: number of rounds to average across
    - heuristics: [list of problem solvers]
    - use_iterative_deepening: boolean of whether to use ID the optimal

    Outputs:
    - accuracy_rate: proportion of the time the optimal is found
    - revenue_deviation: average revenue difference
    '''
    if heuristics is None: raise Exception('compare_optimal: no heuristics provided')

    inaccuracies = np.zeros(len(heuristics))
    revenue_deviation = np.zeros(len(heuristics))
    total_revenue = 0

    for _ in range(rounds):
        try:
            pi = generate_random_pi(m, n)
            ps = ProblemSolver(pi)

            if use_iterative_deepening:
                optimal_policy = IterativeDeepening(pi)
            else:
                optimal_policy = ps.optimalPolicy()

            rev_opt = pi.expectedRevenue(optimal_policy)
            total_revenue += rev_opt

            for i, h in enumerate(heuristics):
                h_policy = h(pi)
                rev_diff = rev_opt - pi.expectedRevenue(h_policy)
                revenue_deviation[i] += rev_diff
                if rev_diff > 0: inaccuracies[i] += 1
        except Exception:
            print(traceback.format_exc())
            print(pi)
            print(IterativeDeepening(pi))
            return

    accuracy_rate = (rounds - inaccuracies) / rounds
    revenue_deviation /= total_revenue
    return accuracy_rate, revenue_deviation