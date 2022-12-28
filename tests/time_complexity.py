'''
Dawson Ren, 12/27/22

Check the time complexity of crucial operations.
'''
import time
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from ProblemInstance.ProblemInstance import ProblemInstance
from ProblemSolver.ProblemSolver import ProblemSolver

ER_SIZES = np.array([10, 50, 100, 250, 500])
OPT_SIZES = np.array([3, 4, 5, 6, 7, 8, 9, 10])
REPEATS = 10
LOOPS = 5

def test_function(title: str, func, sizes: List[int]) -> None:
    avg_times = []

    # get times, loop LOOPS number of times
    for size in sizes:
        times = []

        for _ in range(LOOPS):
            m = n = size

            start = time.perf_counter()

            P = np.random.rand(m, n)
            Q = np.random.rand(m, n)
            R = np.random.rand(m, n) * 10
            pi = ProblemInstance(P, Q, R)

            for _ in range(REPEATS):
                Y = np.random.randint(0, 2, size=(m, n))
                func(pi, Y)

            end = time.perf_counter()
            times.append(end - start)

        # record average times
        avg_times.append(sum(times) / len(times))

    # display
    plt.title(title)
    plt.xlabel('Policy matrix size (n x n)')
    plt.ylabel('Time')
    plt.plot(sizes, avg_times)
    plt.show()

def test_expected_revenue():
    def func(pi: ProblemInstance, Y: np.ndarray) -> None:
        pi.expectedRevenue(Y)
    test_function('Expected Revenue Calculation Time Complexity', func, ER_SIZES)

def test_optimal_solution():
    def func(pi: ProblemInstance, Y: np.ndarray) -> None:
        ps = ProblemSolver(pi)
        ps.optimalPolicy()
    test_function('Optimal Policy Calculation Time Complexity', func, OPT_SIZES)

if __name__ == '__main__':
    er_resp = input('Would you like to show time complexity for getting the expected revenue (Y/N)? ')
    opt_resp = input('Would you like to show time complexity for getting the optimal solution (Y/N)? ')

    if er_resp.lower() == 'y':
        test_expected_revenue()

    if opt_resp.lower() == 'y':
        test_optimal_solution()