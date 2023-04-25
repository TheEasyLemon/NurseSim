'''
Utlity to generate a random ProblemInstance

Dawson Ren
2/6/23
'''
import numpy as np

from ProblemInstance.ProblemInstance import ProblemInstance

def generate_random_pi(m: int, n: int, ones=False, cython=True, round=None, constant_revenue_per_shift=False) -> ProblemInstance:
    '''
    m: natural, the number of nurses
    n: natural, the number of shifts
    ones: boolean, whether N is ones (solvable in polynomial time)
    cython: boolean, use Cython availability function (speedup by around 3x)
    round: natural | None, number of places to round, if None then don't round
    constant_revenue_per_shift: have the same revenue for every nurse in a shift
    '''
    # generates a random ProblemInstance.
    P = np.random.rand(m, n)
    if round is not None: P = P.round(round)
    Q = np.random.rand(m, n)
    if round is not None: Q = Q.round(round)
    R = np.random.rand(m, n) * 10
    if round is not None: R = R.round(round)
    if constant_revenue_per_shift: R = R[0, :]
    N = np.ones(n, dtype=np.int64) if ones else np.random.randint(1, m, size=(n, ), dtype=np.int64)
    return ProblemInstance(P, Q, R, N, cython)