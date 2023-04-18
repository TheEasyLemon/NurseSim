'''
Numerically verify Theorems 3.3 and 3.6.

Try out Mehrotra's suggestion on ordering.
'''
import numpy as np

from ProblemSolver.ProblemSolver import ProblemSolver
from ProblemReader.ProblemReader import loadProblemInstance
from utils.random_pi import generate_random_pi

def partition(lst):
    '''
    Return a list of lists with contiguous members.
    ex. lst = [0, 1, 2, 4, 5, 7]
    returns [[0, 1, 2], [4, 5], [7]]
    '''
    lsts = []
    cur_lst = []
    for i in sorted(lst):
        if len(cur_lst) == 0 or cur_lst[-1] + 1 == i:
            cur_lst.append(i)
        else:
            lsts.append(cur_lst)
            cur_lst = [i]
    if len(cur_lst) != 0: lsts.append(cur_lst)

    return lsts

def theorem_3_3(m=10):
    '''
    Quicker elimination of nurses with runs. In other words,
    eliminate nurses 

    Given a policy vector Y with m nurses and just 1 shift, let
    \delta_k be the change in expected revenue when the kth
    nurse flips from zero to one. Let l be any component of Y
    that is zero, and require that there are only zero components
    between the lth and kth component. Then, \delta_k > \delta_l
    if one of the following holds:
    1. p_k > p_l and q_k > q_l
    2. p_k < p_l and p_k q_k > p_l q_l
    '''
    for _ in range(1):
        print('start new')
        pi = loadProblemInstance('experiments/theorem3_3_counterexample.txt') # generate_random_pi(m, 1, round=2)
        ps = ProblemSolver(pi)
        dyn = ps.optimalColumn(0) # np.zeros(pi.m, dtype=np.int64)
        print(dyn)
        available_nurses = np.arange(pi.m, dtype=np.int64)[dyn.ravel() == 0]
        print('avail nurses:', available_nurses, pi.m)

        for lst in partition(available_nurses):
            if len(lst) == 1: continue

            def turn_on_nurse_revenue(i):
                turn_on = dyn.ravel().copy()
                turn_on[i] = 1
                print(i, turn_on, pi.expectedRevenueCol(0, turn_on))
                return pi.expectedRevenueCol(0, turn_on)

            # get all successive pairs
            for i, k in enumerate(lst):
                for l in lst[i + 1:]:
                    # check the condition
                    if (pi.P[k, 0] > pi.P[l, 0] and pi.Q[k, 0] > pi.Q[l, 0]) or \
                       (pi.P[k, 0] < pi.P[l ,0] and pi.P[k, 0] * pi.Q[k, 0] > pi.P[l, 0] * pi.Q[l, 0]):
                        # check for violations of the expected result
                        if turn_on_nurse_revenue(k) < turn_on_nurse_revenue(l):
                            print(k, l)
                            print(pi)
                            raise Exception('Failed to meet criteria.')


def theorem_3_6():
    '''
    Early stopping condition for checks.

    If the value of adding a nurse (flip from zero to one)
    is less than...? Not sure if I get this one
    '''
    pass

if __name__ == '__main__':
    theorem_3_3(4)

    # pi = generate_random_pi(3, 1, round=1)
    # print(pi)
    # ps = ProblemSolver(pi)
    # print(ps.optimalPolicy())