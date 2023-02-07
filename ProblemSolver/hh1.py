'''
HH1 Algorithm created by Hedieh Sazvar
Refactored by Dawson Ren
2/5/23
'''
import numpy as np

from ProblemInstance.ProblemInstance import ProblemInstance
from ProblemSolver.ProblemSolver import ProblemSolver


def HH1(pi: ProblemInstance):
    # create ProblemSolver for access to N_j = 1 solutions
    ps = ProblemSolver(pi)

    # best policy that we return
    hh1_policy = np.zeros((pi.m, pi.n), dtype=np.int64)

    # do this for every single shift
    for j in range(pi.n):
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
            k = (N_cur & best_policy).argmax() # returns first that we haven't dropped, all guaranteed to be 0 or 1
            N_cur[k] = 0
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
            # an alternative policy is:
            # - add one of the additional_nurses
            # - add nurses from indices 0 to i for i in len(additional_nurses)
            best_additional_policy = None
            best_additional_revenue = 0

            # search through adding only one from additional_nurses
            for i in range(len(additional_nurses)):
                y_alt = np.copy(y_1)
                y_alt[i] = 1
                y_alt_rev = pi.expectedRevenueCol(j, y_alt)
                if y_alt_rev > best_additional_revenue:
                    best_additional_revenue = y_alt_rev
                    best_additional_policy = y_alt

            # search through adding nurses 0 through i
            for i in range(len(additional_nurses)):
                y_alt = np.copy(y_1)
                y_alt[additional_nurses[:i + 1]] = 1
                y_alt_rev = pi.expectedRevenueCol(j, y_alt)
                if y_alt_rev > best_additional_revenue:
                    best_additional_revenue = y_alt_rev
                    best_additional_policy = y_alt

            # Step 2.4 - update solution and parameters
            # update best policy between additional and current best_policy
            best_policy = max(best_additional_policy, best_policy, key=lambda y: pi.expectedRevenueCol(j, y) if y is not None else 0)
            # update n_cur if we found some improvements
            if len(additional_nurses) > 0:
                n_cur += 1
            
            
        # Step 2.5 - return best solution
        hh1_policy[:, j] = best_policy.reshape(best_policy.size)

    return hh1_policy
