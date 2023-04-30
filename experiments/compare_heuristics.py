'''
Finds counterexamples to the HH1 algorithm.

Dawson Ren
2/6/23
'''
import numpy as np
import matplotlib.pyplot as plt

from ProblemSolver.mip import MIP

from utils.comparisons import create_heuristic_comparison_graph, show_heuristic_comparison_graphs

MIP_BinaryY = lambda pi: MIP(pi, binary_y=True)
MIP_ContinuousY = lambda pi: MIP(pi, binary_y=False)

HEURISTIC_NAMES = ['MIP-BinaryY', 'MIP-ContinuousY']
HEURISTICS = [MIP_BinaryY, MIP_ContinuousY]
FILENAME = 'mip'

if __name__ == '__main__':
    create_heuristic_comparison_graph(3, 6, 100, HEURISTICS, HEURISTIC_NAMES, FILENAME, verbose=True)
    show_heuristic_comparison_graphs(HEURISTIC_NAMES, FILENAME)
