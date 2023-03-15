'''
Simple speed comparison for Cython.
'''
import time

import numpy as np

from utils.random_pi import generate_random_pi
from ProblemSolver.ProblemSolver import ProblemSolver

def test_normal_speed(N, size):
    start = time.perf_counter()

    for _ in range(N):
        pi = generate_random_pi(size, 1, cython=False)
        ps = ProblemSolver(pi)

        Y = ps.optimalColumn(0)

    end = time.perf_counter()
    return end - start

def test_cython_speed(N, size):
    start = time.perf_counter()

    for _ in range(N):
        pi = generate_random_pi(size, 1, cython=True)
        ps = ProblemSolver(pi)

        Y = ps.optimalColumn(0)

    end = time.perf_counter()
    
    return end - start

if __name__ == '__main__':
    N = 200
    size = 10

    print(f'normal took {test_normal_speed(N, size)} seconds.')

    # 200 replications of 10x1 takes ~1 second.
    # 11x1 takes around 3.6 seconds.
    # 12x1 takes around 11 seconds.
    print(f'cython took {test_cython_speed(N, size)} seconds.')