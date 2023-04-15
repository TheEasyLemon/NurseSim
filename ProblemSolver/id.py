'''
Iterative Deepening.

Not optimal, slightly suboptimal. Found counter example, see experiments/id_counterexample.txt.

1. Find optimal policy with N_j = 1.
2. Consider all combinations to add to the policy for N_j = n.
   We evaluate the goodness of these additions with the revenue
   FOR THAT n_j, NOT for the largest N_j. We use expectedRevenuColSpecifyN.
3. Repeat step 2 until we hit n = largest N_j.
'''

import numpy as np

from ProblemInstance.ProblemInstance import ProblemInstance
from ProblemSolver.ProblemSolver import ProblemSolver
from utils.col_aggregator import col_aggregator

def IterativeDeepeningCol(pi: ProblemInstance, j: int):
    ps = ProblemSolver(pi)
    # optimal policy with N_j = 1
    best = ps.dynamicColumn(j).reshape(pi.m)

    # all nurses from k are used in k + 1, exhaustively check the unused nurses
    for n in range(2, pi.N[j] + 1):
        inner_best = best
        inner_best_rev = pi.expectedRevenueColSpecifyN(j, best, n)

        free_nurses = pi.m - best.sum()
        if free_nurses == 0: return best.reshape((pi.m, 1))

        # find the best of new policies
        for i in range(2 ** free_nurses):
            # convert i from decimal to numpy array of 0/1s
            y_partial = np.array([[int(k) for k in '{0:b}'.format(i).zfill(free_nurses)]]).reshape(free_nurses)
            y = best.copy()
            y[best == 0] = y_partial

            if (new_rev := pi.expectedRevenueColSpecifyN(j, y, n)) > inner_best_rev:
                inner_best = y
                inner_best_rev = new_rev
                
        best = inner_best

    return best.reshape((pi.m, 1))


IterativeDeepening = col_aggregator(IterativeDeepeningCol)