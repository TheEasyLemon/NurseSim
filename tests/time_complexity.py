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
from ProblemSolver.hh1 import HH1
from utils.random_pi import generate_random_pi

# nxn problem size for expected revenue
ER_SIZES = np.array([3, 5, 10, 13, 15])
# nxn problem size for optimal solution
OPT_SIZES = np.array([3, 4, 5, 6, 7, 8, 9, 10])
# nxn problem size for optimal solution with dynamic programming
DYN_SIZES = np.array([3, 5, 7, 9, 13, 15, 18])
# nxn problem size for optimal solution with hh1 heuristic algorithm
HH1_SIZES = np.array([3, 4, 5, 6, 8, 10, 12, 14, 16])
# how many policies we test with a given ProblemInstance
REPEATS = 5
# how many ProblemInstances we test
LOOPS = 3

def test_function(title: str, func, sizes: List[int], ones=False) -> None:
    avg_times = []

    # get times, loop LOOPS number of times
    for size in sizes:
        times = []

        for _ in range(LOOPS):
            m = n = size

            start = time.perf_counter()

            pi = generate_random_pi(m, n, Nj=1 if ones else None)

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

def test_dynamic_solution():
    def func(pi: ProblemInstance, Y: np.ndarray) -> None:
        ps = ProblemSolver(pi)
        ps.dynamicPolicy()
    test_function('Optimal Dynamic Policy Calculation Time Complexity', func, DYN_SIZES, ones=True)

def test_hh1_solution():
    def func(pi: ProblemInstance, Y: np.ndarray) -> None:
        HH1(pi)
    test_function('Optimal HH1 Calculation Time Complexity', func, HH1_SIZES)

if __name__ == '__main__':
    er_resp = input('Would you like to show time complexity for getting the expected revenue (Y/N)? ')
    opt_resp = input('Would you like to show time complexity for getting the optimal solution by brute force search (Y/N)? ')
    dyn_resp = input('Would you like to show time complexity for getting the optimal solution using dynamic programming (N_j = 1) (Y/N)? ')
    hh1_resp = input('Would you like to show time complexity for getting a heuristic solution using HH1 (Y/N)? ')

    if er_resp.lower() == 'y':
        test_expected_revenue()

    if opt_resp.lower() == 'y':
        test_optimal_solution()

    if dyn_resp.lower() == 'y':
        test_dynamic_solution()

    if hh1_resp.lower() == 'y':
        test_hh1_solution()