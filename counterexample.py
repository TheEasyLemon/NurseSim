'''
Finds counterexamples to the HH1 algorithm.

Dawson Ren
2/6/23
'''
import numpy as np
import matplotlib.pyplot as plt

from ProblemSolver.hh1 import HH1, HH2
from ProblemSolver.hm1 import HM1

from utils.comparisons import compare_to_optimal

HEURISTIC_NAMES = ['HH1', 'HH2', 'HM1']
HEURISTICS = [HH1, HH2, HM1]


def run_heuristic_comparison_graph(low: int, high: int, N: int, verbose=False):
    size = np.arange(low, high + 1)
    size = size.reshape(size.size, 1)
    acc_rate = np.zeros((size.size, len(HEURISTICS)))
    rev_dev = np.zeros((size.size, len(HEURISTICS)))

    for i, s in enumerate(size):
        if verbose: print(f'Running {N} replications for size {int(s)}...')

        accuracy_rate, revenue_deviation = compare_to_optimal(int(s), int(s), N, HEURISTICS, use_iterative_deepening=True)
        
        if verbose:
            for j, h_name in enumerate(HEURISTIC_NAMES):
                print(f'Average accuracy rate for {h_name}: {accuracy_rate[j]}')
                print(f'Average revenue deviation for {h_name}: {revenue_deviation[j]}')
        
        acc_rate[i, :] = accuracy_rate
        rev_dev[i, :] = revenue_deviation

    np.savetxt('hh1_hh2_hm1_acc_rate.csv', np.hstack((size, acc_rate)))
    np.savetxt('hh1_hh2_hm1_rev_dev.csv', np.hstack((size, rev_dev)))


def display_heuristic_comparison_graph_accuracy_rate():
    data = np.loadtxt('hh1_hh2_hm1_acc_rate.csv')
    size = data[:, 0]
    prop_dev = data[:, 1:]
    
    plt.title('Accuracy Rate vs Problem Size ($n$)')
    plt.ylabel('Accuracy Rate')
    plt.xlabel('Problem Size ($n$)')
    for i, h_name in enumerate(HEURISTIC_NAMES):
        plt.plot(size, prop_dev[:, i], label=h_name)
    plt.legend()
    plt.show()

def display_heuristic_comparison_graph_revenue_deviation():
    data = np.loadtxt('hh1_hh2_hm1_rev_dev.csv')
    size = data[:, 0]
    prop_dev = data[:, 1:]
    
    plt.title('Proportion Deviation from Optimal vs Problem Size ($n$)')
    plt.ylabel('Proportion Deviation from Optimal')
    plt.xlabel('Problem Size ($n$)')
    for i, h_name in enumerate(HEURISTIC_NAMES):
        plt.plot(size, prop_dev[:, i], label=h_name)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    run_heuristic_comparison_graph(4, 10, 200, verbose=True)
    display_heuristic_comparison_graph_accuracy_rate()
    display_heuristic_comparison_graph_revenue_deviation()

