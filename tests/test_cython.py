'''
Simple speed comparison for Cython.
'''
import time

import numpy as np

from utils.random_pi import generate_random_pi
from ProblemSolver.ProblemSolver import ProblemSolver
from utils.monte_carlo import monte_carlo_expected_revenue

def test_random_correctness():
    num = 10

    for _ in range(num):
        pi = generate_random_pi(8, 8)
        Y = np.random.randint(0, 2, size=(8, 8))
        exact = pi.expectedRevenue(Y)
        mc = monte_carlo_expected_revenue(pi, Y)
        assert abs(exact - mc) < 1

def test_normal(N, size):
    start = time.perf_counter()

    for _ in range(N):
        pi = generate_random_pi(size, size)
        ps = ProblemSolver(pi)

        Y = ps.optimalColumn(0, cython=False)

    end = time.perf_counter()
    return end - start

def test_cython(N, size):
    start = time.perf_counter()

    for _ in range(N):
        pi = generate_random_pi(size, size)
        ps = ProblemSolver(pi)

        Y_heuristic = ps.optimalColumnHeuristic(0)

    end = time.perf_counter()
    
    return end - start

if __name__ == '__main__':
    N = 200
    size = 12

    # 200 replications of 10x1 takes ~2 seconds.
    # 11x1 takes around 5.5 seconds.
    # 12x1 takes around 13 seconds
    print(f'cython took {test_cython(N, size)} seconds.')

    # test_random_correctness()