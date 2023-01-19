'''
Provide Monte Carlo estimation methods for expected revenue.

Dawson Ren
January 6th, 2022
'''
import math

import numpy as np

from ProblemInstance.ProblemInstance import ProblemInstance

def hoeffding_bound(e: float, alpha: float) -> int:
    # provides the number of samples needed in MC simulation to give error
    # +/- e*h (h is supremum of outputs of the estimation) with probability 1 - alpha.
    return int((math.log(2 / alpha)) / (2 * e ** 2)) + 1

def limit_shifts(sample: np.ndarray, N: np.ndarray):
    # limit the number of shifts that can be taken by nurses in FCFS order
    # ex.
    # [[0, 1, 1]
    #  [1, 1, 1]
    #  [1, 1, 0]]
    # N = [1, 2, 1]
    # results in...
    # [[0, 1, 1]
    #  [1, 1, 0]
    #  [0, 0, 0]]

    # is there a way to vectorize or use native Numpy code? Feels slow
    for j in range(sample.shape[1]):
        n = 0

        for i in range(sample.shape[0]):
            if n >= N[j]:
                sample[i:, j] = 0
                break
            if sample[i, j] == 1:
                n += 1

    return sample

def monte_carlo_expected_revenue(pi: ProblemInstance, Y: np.ndarray, alpha: float = 0.01, e: float = 0.005) -> float:
    h = pi.R.sum() # maximum revenue we can receive
    m, n = pi.P.shape

    N = hoeffding_bound(e, alpha)

    print(f'Beginning Monte Carlo Expected Revenue simulation with {N} iterations on a ProblemInstance.')

    estimate = 0

    for _ in range(N):
        # sample from P
        P_sample = (np.random.rand(m, n) < pi.P).astype(np.int64)
        # sample from Q
        Q_sample = (np.random.rand(m, n) < pi.Q).astype(np.int64)
        # combine with Y to see which shifts nurses are available for and which visible to them
        PY_sample = P_sample * Y
        # limit shifts to N
        PY_sample = limit_shifts(PY_sample, pi.N)
        # combine with Q to get which shifts fulfilled
        fulfilled = PY_sample * Q_sample
        # print(P_sample, Q_sample, PY_sample, fulfilled, sep='\n')
        # combine with R to get sample revenue
        estimate += (fulfilled * pi.R).sum()

    estimate /= N

    print(f'Estimated to be within [{estimate} +/- {e * h}] with {(1 - alpha) * 100}% confidence, exact method predicts {pi.expectedRevenue(Y)}')



