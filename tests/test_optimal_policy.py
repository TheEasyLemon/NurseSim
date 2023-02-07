'''
Testing script that uses Monte Carlo estimation to find the revenue of an
optimal policy. We do this by generating all possible policies (col by col),
finding the optimal col, and aggregating the columns together.

We test this against the formula we use to find the expected value of a
policy with N_j >= 1. We confirm that our formula is correct by finding
another policy that's just as good as the one we found with Monte Carlo
simulation.
'''
import numpy as np
from typing import Generator

from ProblemInstance.ProblemInstance import ProblemInstance
from ProblemSolver.ProblemSolver import ProblemSolver

from utils.monte_carlo import monte_carlo_expected_revenue_col
from utils.random_pi import generate_random_pi

def generate_all_policies_col(m: int) -> Generator[np.ndarray, None, None]:
    # Generator for possible policies
    for i in range(2 ** (m - 1)):
        # convert i from decimal to numpy array of 0/1s
        yield np.array([[int(k) for k in '{0:b}'.format((i << 1) + 1).zfill(m)]]).reshape((m, ))

def monte_carlo_optimal_policy_col(pi: ProblemInstance, shift: int):
    best_policy = None
    best_revenue = 0

    for col_policy in generate_all_policies_col(pi.m):
        # print(f'Trying policy {col_policy}...')
        revenue = monte_carlo_expected_revenue_col(pi, col_policy, shift)
        if revenue > best_revenue:
            best_policy = col_policy
            best_revenue = revenue

    return best_policy.reshape((pi.m, 1))

def formula_optimal_policy_col(pi: ProblemInstance, shift: int):
    ps = ProblemSolver(pi)
    return ps.optimalColumn(shift)

def col_aggregator(col_func: any):
    def optimal(pi: ProblemInstance):
        return np.hstack([col_func(pi, i) for i in range(pi.n)])
    return optimal

monte_carlo_optimal_policy = col_aggregator(monte_carlo_optimal_policy_col)
formula_optimal_policy = col_aggregator(formula_optimal_policy_col)

if __name__ == '__main__':
    pi = generate_random_pi(8, 8)
    # print(monte_carlo_optimal_policy(pi))
    print(formula_optimal_policy(pi))

