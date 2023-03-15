'''
HH1 Algorithm created by Hedieh Sazvar
Refactored by Dawson Ren
2/5/23

Variations on HH1
1. Only add one single nurse in each iteration (HH1SingCom)
2. Same as HH1, only use highest value nurse (v=rpq), (HH1MaxRev)
3. Remove all nurses, not just the first/most valuable one (HH1RemAll)
4. Same as HH1, only check from last nurse (HH1NC)
'''
from typing import Callable, List
from functools import partial

import numpy as np

from ProblemInstance.ProblemInstance import ProblemInstance
from ProblemSolver.ProblemSolver import ProblemSolver
from utils.col_aggregator import col_aggregator

###
### Removers
###

def remove_first_nurse(N_cur: np.ndarray, best_policy: np.ndarray, pi: ProblemInstance, j: int) -> np.ndarray:
    '''
    Return a copy of N_cur with the first nurse removed that's also seen in best_policy.

    Used for HH1, HH1SingCom, HH1NC
    '''
    k = (N_cur & best_policy).argmax() # returns first that we haven't dropped, all guaranteed to be 0 or 1
    N_cur_copy = N_cur.copy()
    N_cur_copy[k] = 0
    return N_cur_copy

def remove_most_valuable_nurse(N_cur: np.ndarray, best_policy: np.ndarray, pi: ProblemInstance, j: int) -> np.ndarray:
    '''
    Return a copy of N_cur with the most valuable nurse removed that's also seen in best_policy.

    Used for HH1MaxRev
    '''
    removable = np.where(N_cur & best_policy, pi.V[:, j], 0)
    # if all zeros, then revert back to remove first
    if np.all(removable == 0):
        remove_first_nurse(N_cur, best_policy, pi, j)
        return
    k = removable.argmax() # returns most valuable nurse
    N_cur_copy = N_cur.copy()
    N_cur_copy[k] = 0 # remove nurse from considered nurses
    return N_cur_copy

def remove_all_nurses(N_cur: np.ndarray, best_policy: np.ndarray, pi: ProblemInstance, j: int) -> np.ndarray:
    '''
    Return a copy of N_cur with all nurses removed that's also seen in best_policy.

    Used for HH1RemAll
    '''
    N_cur_copy = N_cur.copy()
    N_cur_copy = N_cur & ~best_policy
    return N_cur_copy

###
### Modifiers
###

def set_one_nurse_on(y_alt: np.ndarray, i: int, additional_nurses: np.ndarray) -> np.ndarray:
    '''
    Try adding each available nurse one at a time.
    
    Used for HH1, HH1SingCom, HH1MaxRev, HH1NC, HH1RemAll
    '''
    y_alt[i] = 1

def set_nurse_ascending_on(y_alt: np.ndarray, i: int, additional_nurses: np.ndarray) -> np.ndarray:
    '''
    Try adding each available nurse one at a time, or including 0 to i.

    Used for HH1, HH1MaxRev, HH1RemAll
    '''
    y_alt[additional_nurses[:i + 1]] = 1

def set_nurse_descending_on(y_alt: np.ndarray, i: int, additional_nurses: np.ndarray) -> np.ndarray:
    '''
    Try adding each available nurse one at a time, or including i to end.

    Used for HH1NC
    '''
    y_alt[additional_nurses[i:]] = 1

###
### Helper functions
### 

def search_policies(additional_nurses: np.ndarray, y_1: np.ndarray, pi: ProblemInstance, j: int,
                    modifiers: List[Callable[[np.ndarray, int], np.ndarray]]) -> np.ndarray:
    '''
    Search all alternative policies
    '''
    best_additional_policy = None
    best_additional_revenue = -1

    # search through policies
    for modify in modifiers:
        for i in range(len(additional_nurses)):
            y_alt = np.copy(y_1)
            modify(y_alt, i, additional_nurses)
            y_alt_rev = pi.expectedRevenueCol(j, y_alt)
            if y_alt_rev > best_additional_revenue:
                best_additional_revenue = y_alt_rev
                best_additional_policy = y_alt

    return best_additional_policy

def GenericHH1Col(pi: ProblemInstance, j: int, remover: Callable[[np.ndarray, np.ndarray], np.ndarray],
                  modifiers: List[Callable[[np.ndarray, int], np.ndarray]]) -> np.ndarray:
    '''
    Generic driver to find an HH1-style solution for a column.
    '''
    # create ProblemSolver for access to N_j = 1 solutions
    ps = ProblemSolver(pi)

    # Step 1.1: get the best policy for N_j = 1
    y_1 = ps.dynamicColumn(j).reshape(pi.m)
    best_policy = y_1 # best so far

    # Step 1.2: loop initialization
    N_cur = np.ones(pi.m, dtype=np.int64) # set of all nurse indices we're considering
    n_cur = 2 # the n_j we currently consider

    # Step 2.1: Stopping Criteria
    # only do Steps 2.2 - 2.4 if
    # we have N_j >= 1 and at least
    # one nurse scheduled and we have
    # at least one nurse left and
    # our current n_cur is less than N_j
    while pi.N[j] >= 1 and y_1.sum() != 0 and N_cur.sum() >= 1 and n_cur <= pi.N[j]:
        # Step 2.2
        # Step 2.2.1: find first nurse we offer
        # shift j to, call it k, and remove from N_cur
        N_cur = remover(N_cur, best_policy, pi, j)
        active_indices = np.nonzero(N_cur)[0] # gets all indices where nurses available
        if active_indices.size == 0: break # if no more nurses to drop, break

        # Step 2.2.2: find solution to modified problem
        y_mod = ps.dynamicColumn(j, active_indices)

        # Step 2.2.3: find additional nurses considered compared to original N_j = 1 solution
        # insert back into shift to make same length
        y_m = np.zeros(pi.m, dtype=np.int64)
        y_m[active_indices] = y_mod.reshape(y_mod.size)
        # contains the indices of the nurses we should add now
        additional_nurses = np.nonzero(y_m & ~best_policy)[0]

        # Step 2.3
        # Step 2.3.1: define alternative policies and find the best one
        best_additional_policy = search_policies(additional_nurses, y_1, pi, j, modifiers)

        # Step 2.4 - update solution and parameters
        # update best policy between additional and current best_policy
        best_policy = max(best_additional_policy, best_policy, key=lambda y: 0 if y is None else pi.expectedRevenueCol(j, y))
        
        # update n_cur if we found some improvements
        if len(additional_nurses) > 0:
            n_cur += 1

    return best_policy.reshape(best_policy.size, 1)

def HH2Col(pi: ProblemInstance, j: int):
    hh1 = GenericHH1Col(pi, j, remover=remove_first_nurse, modifiers=[set_one_nurse_on, set_nurse_ascending_on])
    hh1_singcom = GenericHH1Col(pi, j, remover=remove_first_nurse, modifiers=[set_one_nurse_on])
    hh1_maxrev = GenericHH1Col(pi, j, remover=remove_most_valuable_nurse, modifiers=[set_one_nurse_on])
    hh1_remall = GenericHH1Col(pi, j, remover=remove_all_nurses, modifiers=[set_one_nurse_on, set_nurse_ascending_on])
    hh1_nc = GenericHH1Col(pi, j, remover=remove_first_nurse, modifiers=[set_one_nurse_on, set_nurse_descending_on])

    best_policy = max([hh1, hh1_singcom, hh1_maxrev, hh1_remall, hh1_nc], key=lambda y: pi.expectedRevenueCol(j, y.reshape(y.size)))
    return best_policy

###
### Approximate Solvers (pi: ProblemInstance) -> Policy (np.array, m x n)
###

HH1 = col_aggregator(partial(GenericHH1Col, remover=remove_first_nurse, modifiers=[set_one_nurse_on, set_nurse_ascending_on]))
HH2 = col_aggregator(HH2Col)
