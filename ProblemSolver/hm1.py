'''
Hedieh-Mehortra Algorithm 1

1. Find optimal policy with N_j = 1.
2. Consider only combinations of length <= k to add
   to the policy for N_j = n_j.
3. Repeat step 2 until no additions result in increased revenue.
'''
from itertools import combinations

import numpy as np

from ProblemInstance.ProblemInstance import ProblemInstance
from ProblemSolver.ProblemSolver import ProblemSolver
from utils.col_aggregator import param_col_aggregator, col_aggregator

def HM1Col(pi: ProblemInstance, j: int, k=None):
    ps = ProblemSolver(pi)
    # optimal policy with N_j = 1
    best = ps.dynamicColumn(j).reshape(pi.m)
    # get available nurses (not scheduled)
    available_nurses = np.arange(best.size)[best == 0]
    # loop termination condition
    better_solutions_exist = True

    candidates_checked = 0

    while better_solutions_exist:
        # reset available nurses
        available_nurses = np.arange(best.size)[best == 0]

        # try new inner best
        inner_best = best.copy()
        # print('Starting loop with', inner_best)

        better_solutions_exist = False
        max_seq_len = k if k is not None else (inner_best == 0).sum()

        # find the best out of all sequences k
        for seq_len in range(1, max_seq_len + 1):
            for combo in combinations(available_nurses, seq_len):
                candidate = best.copy()
                candidate[np.array(combo)] = 1
                # print('trying out', candidate)
                if pi.expectedRevenueCol(j, inner_best) < pi.expectedRevenueCol(j, candidate):
                    candidates_checked += 1
                    # print('it was better')
                    inner_best = candidate
                    better_solutions_exist = True
                    
        best = inner_best.copy()

    return best.reshape((pi.m, 1))

def HM1Col_blank_slate(pi: ProblemInstance, j: int, k=None):
    ps = ProblemSolver(pi)
    # empty policy
    best = np.zeros(pi.m, dtype=np.int64)
    # get available nurses (not scheduled)
    available_nurses = np.arange(best.size)[best == 0]
    # loop termination condition
    better_solutions_exist = True

    candidates_checked = 0

    while better_solutions_exist:
        # reset available nurses
        available_nurses = np.arange(best.size)[best == 0]

        # try new inner best
        inner_best = best.copy()
        # print('Starting loop with', inner_best)

        better_solutions_exist = False
        max_seq_len = k if k is not None else (inner_best == 0).sum()

        # find the best out of all sequences k
        for seq_len in range(1, max_seq_len + 1):
            for combo in combinations(available_nurses, seq_len):
                candidate = best.copy()
                candidate[np.array(combo)] = 1
                # print('trying out', candidate)
                if pi.expectedRevenueCol(j, inner_best) < pi.expectedRevenueCol(j, candidate):
                    candidates_checked += 1
                    # print('it was better')
                    inner_best = candidate
                    better_solutions_exist = True
                    
        best = inner_best.copy()

    return best.reshape((pi.m, 1))


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
HM1 = param_col_aggregator(HM1Col)
HM1_blank_slate = param_col_aggregator(HM1Col_blank_slate)