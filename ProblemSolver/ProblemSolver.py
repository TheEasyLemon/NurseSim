'''
Provides the ProblemSolver class, finding the optimal policy for a given ProblemInstance.
'''
import numpy as np
from typing import Generator

from ProblemInstance.ProblemInstance import ProblemInstance

class ProblemSolver:
    def __init__(self, prob: ProblemInstance) -> None:
        self.prob = prob

    def _getAllPolicies(self, m, n) -> Generator[np.ndarray, None, None]:
        # exponentially large number of policies...
        for i in range(2 ** (m * n)):
            yield np.array([int(k) for k in '{0:b}'.format(i).zfill(m * n)]).reshape(m, n)

    def _getFeasiblePolicies(self, m, n) -> Generator[np.ndarray, None, None]:
        for i in range(2 ** (m * n)):
            Y = np.array([int(k) for k in '{0:b}'.format(i).zfill(m * n)]).reshape(m, n)

            if m >= n:
                # Y should have at least a 1 in every column
                if (Y.sum(axis=0) >= 0).sum() >= n:
                    yield Y
            else:
                # Y should be scheduling every nurse
                if (Y.sum(axis=1) >= 0).sum() >= m:
                    yield Y

    def bruteForceOptimalPolicy(self, optimize=False) -> np.ndarray:
        best_rev = 0
        best_pol = None
        gen = self._getFeasiblePolicies(self.prob.m, self.prob.n) if optimize else self._getAllPolicies(self.prob.m, self.prob.n)

        for pol in gen:
            if (new_rev := self.prob.expectedRevenue(pol)) > best_rev:
                best_rev = new_rev
                best_pol = pol

        if best_pol is None:
            raise Exception('ProblemSolver: No feasible policies')

        return best_pol

    def optimalColumn(self, shift: int) -> np.ndarray:
        best_rev = 0
        best_col = None
        m, n = self.prob.m, self.prob.n

        # speedup: it's always advantageous to schedule the last nurse, assuming all revenue is non-negative
        # use bitshifting to accomplish this
        for i in range(2 ** (m - 1)):
            # convert i from decimal to numpy array of 0/1s
            y = np.array([[int(k) for k in '{0:b}'.format((i << 1) + 1).zfill(m)]]).T

            if (new_rev := self.prob.expectedRevenueCol(shift, y)) > best_rev:
                best_rev = new_rev
                best_col = y

        if best_col is None:
            raise Exception('ProblemSolver: No feasible policies')

        return best_col

    def optimalPolicy(self) -> np.ndarray:
        return np.hstack([self.optimalColumn(i) for i in range(self.prob.n)])
