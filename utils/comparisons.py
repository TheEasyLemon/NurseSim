'''
Compare problem solvers/heuristics.
'''
from typing import Tuple, List
import traceback
import sys

import numpy as np
import matplotlib.pyplot as plt

from ProblemSolver.id import IterativeDeepening
from ProblemSolver.ProblemSolver import ProblemSolver
from utils.random_pi import generate_random_pi

def compare_to_optimal(m: int, n: int, rounds: int, heuristics=None, use_iterative_deepening=False) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Inputs:
    - m: number of nurses
    - n: number of shifts
    - rounds: number of rounds to average across
    - heuristics: [list of problem solvers]
    ** DEPRECATED **
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
                # DEPRECATED: IterativeDeepening has a counterexample where it doesn't return optimal
                optimal_policy = IterativeDeepening(pi)
            else:
                optimal_policy = ps.optimalPolicy()

            rev_opt = pi.expectedRevenue(optimal_policy)
            total_revenue += rev_opt

            for i, h in enumerate(heuristics):
                h_policy = h(pi)
                rev_heur = pi.expectedRevenue(h_policy)
                rev_diff = rev_opt - rev_heur
                revenue_deviation[i] += rev_diff
                if rev_diff > sys.float_info.epsilon: inaccuracies[i] += 1
                if rev_diff < 0:
                    print('OPTIMAL IS NOT OPTIMAL:', rev_opt, rev_heur, optimal_policy, h_policy, pi, sep='\n')
                    raise Exception('OptimalityException')
        except Exception:
            print(traceback.format_exc())
            return

    accuracy_rate = (rounds - inaccuracies) / rounds
    revenue_deviation /= total_revenue
    return accuracy_rate, revenue_deviation


def create_heuristic_comparison_graph(low: int, high: int, N: int, heuristics: List[ProblemSolver],
                                   heuristic_names: List[str], filename: str, verbose=False, save_txt=True):
    '''
    low: the smallest problem size to solve, inclusive (with problem size = number of nurses (m) = number of shifts (n))
    high: the largest problem size to solve, inclusive
    N: the number of replications
    heuristics: a list of ProblemSolvers to compare to optimal
    heuristic_names: a list of the solver's names
    filename: the beginning of the filename for the resulting text files
    verbose: produce output by heuristic name?
    save_txt: save output of comparison to a txt file?
    '''
    size = np.arange(low, high + 1)
    size = size.reshape(size.size, 1)
    acc_rate = np.zeros((size.size, len(heuristics)))
    rev_dev = np.zeros((size.size, len(heuristics)))

    for i, s in enumerate(size):
        if verbose: print(f'Running {N} replications for size {int(s)}...')

        accuracy_rate, revenue_deviation = compare_to_optimal(int(s), 1, N, heuristics)
        
        if verbose:
            for j, h_name in enumerate(heuristic_names):
                print(f'Average accuracy rate for {h_name}: {accuracy_rate[j]}')
                print(f'Average revenue deviation for {h_name}: {revenue_deviation[j]}')
        
        acc_rate[i, :] = accuracy_rate
        rev_dev[i, :] = revenue_deviation

    if save_txt:
        np.savetxt(f'experiments/{filename}_acc_rate.csv', np.hstack((size, acc_rate)))
        np.savetxt(f'experiments/{filename}_rev_dev.csv', np.hstack((size, rev_dev)))


def display_heuristic_comparison_graph_accuracy_rate(heuristic_names: List[str], filename: str):
    data = np.loadtxt(f'experiments/{filename}_acc_rate.csv')
    size = data[:, 0]
    prop_dev = data[:, 1:]
    
    plt.title('Accuracy Rate vs Problem Size ($n$)')
    plt.ylabel('Accuracy Rate')
    plt.xlabel('Problem Size ($n$)')
    for i, h_name in enumerate(heuristic_names):
        plt.plot(size, prop_dev[:, i], label=h_name)
    plt.legend()
    plt.show()

def display_heuristic_comparison_graph_revenue_deviation(heuristic_names: List[str], filename: str):
    data = np.loadtxt(f'experiments/{filename}_rev_dev.csv')
    size = data[:, 0]
    prop_dev = data[:, 1:]
    
    plt.title('Proportion Deviation from Optimal vs Problem Size ($n$)')
    plt.ylabel('Proportion Deviation from Optimal')
    plt.xlabel('Problem Size ($n$)')
    for i, h_name in enumerate(heuristic_names):
        plt.plot(size, prop_dev[:, i], label=h_name)
    plt.legend()
    plt.show()


def show_heuristic_comparison_graphs(heuristic_names, filename):
    display_heuristic_comparison_graph_accuracy_rate(heuristic_names, filename)
    display_heuristic_comparison_graph_revenue_deviation(heuristic_names, filename)