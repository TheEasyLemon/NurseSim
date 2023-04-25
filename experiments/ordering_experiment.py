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

def theorem_3_3(m=10, rounds=1000):
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
    for i in range(1000):
        pi = generate_random_pi(m, 1) # loadProblemInstance('experiments/theorem3_3_counterexample.txt')
        ps = ProblemSolver(pi)
        dyn = ps.optimalColumn(0)
        available_nurses = np.arange(pi.m, dtype=np.int64)[dyn.ravel() == 0]

        for lst in partition(available_nurses):
            if len(lst) == 1: continue

            def turn_on_nurse_revenue(i):
                turn_on = dyn.ravel().copy()
                turn_on[i] = 1
                return pi.expectedRevenueCol(0, turn_on)

            # get all successive pairs
            for i, l in enumerate(lst):
                for k in lst[i + 1:]:
                    # check the condition
                    pk = pi.P[k, 0]
                    pl = pi.P[l, 0]
                    qk = pi.Q[k, 0]
                    ql = pi.Q[l, 0]
                    if (pk > pl and qk > ql) or (pk < pl and pk * qk > pl * ql):
                        k_rev = turn_on_nurse_revenue(k)
                        l_rev = turn_on_nurse_revenue(l)
                        # check for violations of the expected result
                        if k_rev < l_rev:
                            print('Optimal Column for N_j = 1', dyn, sep='\n')
                            print('Available Nurses:', available_nurses)
                            print('l:', l, 'k:', k)
                            print('l rev:', l_rev, 'k rev:', k_rev)
                            print('p_l:', pl, 'q_l', ql, 'p_lq_l', pl * ql)
                            print('p_k:', pk, 'q_k', qk, 'p_kq_k', pk * qk)
                            print(pi)
                            raise Exception(f'Failed to meet criteria on iteration {i}')


def theorem_3_6():
    '''
    Early stopping condition for checks.

    If the value of adding a nurse (flip from zero to one)
    is less than...? Not sure if I get this one
    '''
    pass

if __name__ == '__main__':
    theorem_3_3(m=10)

    # pi = generate_random_pi(3, 1, round=1)
    # print(pi)
    # ps = ProblemSolver(pi)
    # print(ps.optimalPolicy())