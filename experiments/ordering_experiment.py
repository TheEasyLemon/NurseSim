'''
Numerically verify Theorems 3.3 and 3.6.

Try out Mehrotra's suggestion on ordering.
'''
import numpy as np

from ProblemSolver.ProblemSolver import ProblemSolver
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

class PQ:
    '''
    Wrapper that helps with comparisons for theorem 3.3.
    '''
    def __init__(self, p, q):
        self.p = p
        self.q = q
        self.pq = p * q
        print(p, q, p * q)

    def __lt__(self, other):
        return (self.q < other.q) and (self.pq < other.pq)
    
    def __eq__(self, other):
        p_better = self.p > other.p
        q_better = self.q > other.q
        pq_better = self.pq > other.pq
        return ((p_better and not q_better and pq_better) or
                (not p_better and q_better and not pq_better))

def theorem_3_3():
    '''
    Quicker elimination of nurses with runs. In other words,
    eliminate nurses 

    Given a policy vector Y with m nurses and just 1 shift, let
    \delta_k be the change in expected revenue when the kth
    nurse flips from zero to one. Let l be any component of Y
    that is zero, and require that there are only zero components
    between the lth and kth component. Then, \delta_k > \delta_l
    if one of the following holds:
    1. p_k > p_j and q_k > q_j
    2. p_k < p_j and p_k q_k > p_j q_j
    '''
    for _ in range(10):
        pi = generate_random_pi(5, 1, round=2)
        ps = ProblemSolver(pi)
        dyn = ps.dynamicColumn(0)
        available_nurses = np.arange(pi.m, dtype=np.int64)[dyn.ravel() == 0]
        print(available_nurses, pi.m)

        for lst in partition(available_nurses):
            if len(lst) == 1: continue

            indices_ordered_by_pq = sorted(lst, key=lambda i: PQ(pi.P[i, 0], pi.Q[i, 0]))

            def turn_on_nurse_revenue(i):
                turn_on = dyn.ravel().copy()
                turn_on[i] = 1
                print(i, turn_on, pi.expectedRevenueCol(0, turn_on))
                return pi.expectedRevenueCol(0, turn_on)

            indices_ordered_by_rev = sorted(lst, key=turn_on_nurse_revenue, reverse=True)

            print(indices_ordered_by_pq)
            print(indices_ordered_by_rev)

            print(pi)
            
            assert indices_ordered_by_pq == indices_ordered_by_rev


def theorem_3_6():
    '''
    Early stopping condition for checks.

    If the value of adding a nurse (flip from zero to one)
    is less than...? Not sure if I get this one
    '''
    pass

def ordering_experiment():
    '''
    Create a policy. Order the policies by...?
    '''
    

if __name__ == '__main__':
    # theorem_3_3()
    # ordering_experiment()
    pi = generate_random_pi(3, 1, round=1)
    print(pi)
    ps = ProblemSolver(pi)
    print(ps.optimalPolicy())