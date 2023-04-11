'''
Hedieh-Mehortra Algorithm 1

1. Find optimal policy with N_j = 1.
2. Consider only combinations of length <= k to add
   to the policy for N_j = n_j. We evaluate the goodness
   of these additions with the value of the largest N_j.
3. Repeat step 2 until no additions result in increased revenue.
'''
from itertools import combinations

import numpy as np

from ProblemInstance.ProblemInstance import ProblemInstance
from ProblemSolver.ProblemSolver import ProblemSolver
from utils.col_aggregator import col_aggregator


def HM1Col(pi: ProblemInstance, j: int, k=None, blank_slate=False):
    ps = ProblemSolver(pi)
    # optimal policy with N_j = 1
    best = np.zeros(pi.m, dtype=np.int64) if blank_slate else ps.dynamicColumn(j).reshape(pi.m)
    # get available nurses (not scheduled)
    available_nurses = np.arange(best.size)[best == 0]
    # loop termination condition
    better_solutions_exist = True

    while better_solutions_exist:
        # reset available nurses
        available_nurses = np.arange(best.size)[best == 0]

        # try new inner best
        inner_best = best.copy()

        better_solutions_exist = False
        max_seq_len = k if k is not None else (inner_best == 0).sum()

        # find the best out of all sequences k
        for seq_len in range(1, max_seq_len + 1):
            for combo in combinations(available_nurses, seq_len):
                candidate = best.copy()
                candidate[np.array(combo)] = 1
                if pi.expectedRevenueCol(j, inner_best) < pi.expectedRevenueCol(j, candidate):
                    inner_best = candidate
                    better_solutions_exist = True
                    
        best = inner_best.copy()

    return best.reshape((pi.m, 1))


HM1 = col_aggregator(HM1Col)