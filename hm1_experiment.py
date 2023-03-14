'''
Experiments with the HM1 Algorithm.

Experiments:
1. Try to start from a blank slate (empty policy) and see if HM1 produces optimal for N_j = 1

Seems to work for N_j = 1, takes much longer to run though.

2. Try to solve with an iterative deepening approach (Solve for N_j = 1, then 2, then 3...etc.)

This proves that the theorem is correct, however it has a bad worst-case run time. It's possible
that we search through O(N_j 2^n), because we may end up doing an exhaustive search for every
n <= N_j. In practice however, the ID algorithm is faster than brute force search.

The concept of iterative deepening comes from chess and other minimax games with decision
trees. In order to determine which moves are good to a certain search depth, we start with
a decision tree with the minimum depth = 1, then perform alpha-beta pruning, then try with
depth = 2, prune, then depth = 3...etc. The key to this approach is that we repeat very
little work each time we "deepen" our search. This is because alpha-beta pruning is incredibly
efficient at eliminating possible moves.

In this case, we'd hope that we could eliminate some nurses or some combinations of nurses,
which might be very helpful. Let's see!

3. Generate symbolic equations of revenue, see how the equation changes over time
   Every time we add a nurse to the schedule, we guarantee that revenue increases. Is there a pattern?

TODO: Add sympy and do this, limit N_j to be <= 5.

Dawson Ren
3/12/23
'''
from typing import Tuple
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from sympy import init_printing
from sympy.matrices.dense import matrix_multiply_elementwise

from ProblemSolver.hm1 import HM1_blank_slate, IterativeDeepening
from ProblemSolver.ProblemSolver import ProblemSolver
from utils.random_pi import generate_random_pi

def latex_print(expr):
    text_kwargs = dict(ha='center', va='center', fontsize=14, wrap=True)
    plt.plot()
    plt.text(0, 0, rf'${latex(expr)}$', **text_kwargs)
    plt.show()

def experiment_1():
    '''
    Identify a failure of HM1.
    '''
    m = 12
    max_iter = 1

    for i in range(max_iter):
        pi = generate_random_pi(m, m)
        ps = ProblemSolver(pi)
        ps_sol = ps.optimalPolicyHeuristic()
        print('Optimal:\n', ps_sol)
        hm1_sol = HM1_blank_slate(pi)
        print('HM1:\n', hm1_sol)
        if pi.expectedRevenue(hm1_sol) < pi.expectedRevenue(ps_sol):
            print(f'Found inferior after {i} iterations:')
            print(pi)
            print('HM1 Revenue:', pi.expectedRevenue(hm1_sol))
            print('Optimal Revenue:', pi.expectedRevenue(ps_sol))
            return

    print(f'Unable to identify inferiority after {max_iter} iterations.')

def experiment_2():
    '''
    See if Iterative Deepening produces an optimal solution.
    '''
    m = 14
    max_iter = 1

    for i in range(max_iter):
        print('Iteration:', i + 1)
        pi = generate_random_pi(m, m)
        ps = ProblemSolver(pi)
        ps_sol = ps.optimalPolicyHeuristic()
        # print('Optimal:\n', ps_sol)
        hm1_sol = IterativeDeepening(pi)
        print('HM1:\n', hm1_sol)
        if pi.expectedRevenue(hm1_sol) < pi.expectedRevenue(ps_sol):
            print(f'Found inferior after {i} iterations:')
            print(pi)
            print('Iterative Deepening Revenue:', pi.expectedRevenue(hm1_sol))
            print('Optimal Revenue:', pi.expectedRevenue(ps_sol))
            return

def experiment_3():
    m = 5

    # generate the revenue function for a given problem
    pi = generate_random_pi(m, m)
    shift = 0
    N = pi.N[shift]
    N = 1

    # r = v * y * a
    v = Matrix(symbols(f'v0:{m}')) # pi.V[:, shift].reshape(pi.m)
    y = Matrix(symbols(f'y0:{m}'))
    p = Matrix(symbols(f'p0:{m}')) # pi.P[:, shift].reshape(pi.m)
    
    # now, we calculate a
    a = Matrix(np.zeros(pi.m))

    # first N nurses get availability 1
    for i in range(N):
        a[i] = 1

    # for N_j = 1
    for i in range(1, pi.m):
        a[i] += (1 - p[i - 1] * y[i - 1]) * a[i - 1]

    # for N_j = 2 and above
    for n in range(1, N):
        for i in range(n + 1, pi.m):
            for tup in combinations(range(i), n):
                prob = 1
                for k in range(i):
                    if k in tup:
                        prob *= p[k] * y[k]
                    else:
                        prob *= (1 - p[k] * y[k])
                a[i] += prob

    # enforce first N nurses get availability 1
    for i in range(N):
        a[i] = 1

    vy = matrix_multiply_elementwise(v, y)
    r = sum(matrix_multiply_elementwise(a, vy))
    latex_print(r)

if __name__ == '__main__':
    init_printing()
    # experiment_1()
    # experiment_2()
    experiment_3()