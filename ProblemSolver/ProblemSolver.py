'''
Provides the ProblemSolver class, finding the optimal policy for a given ProblemInstance.
'''
import numpy as np
from typing import Generator

from ProblemInstance.ProblemInstance import ProblemInstance

class ProblemSolver:
    def __init__(self, prob: ProblemInstance) -> None:
        self.prob = prob

    def _getFeasiblePolicies(self, m, n) -> Generator[np.ndarray, None, None]:
        # exponentially large number of policies...
        for i in range(2 ** (m * n)):
            yield np.array([int(k) for k in '{0:b}'.format(i).zfill(m * n)]).reshape(m, n)

    def bruteForceOptimalPolicy(self) -> np.ndarray:
        best_rev = 0
        best_pol = None

        for pol in self._getFeasiblePolicies(self.prob.m, self.prob.n):
            if (new_rev := self.prob.expectedRevenue(pol)) > best_rev:
                best_rev = new_rev
                best_pol = pol

        if best_pol is None:
            raise Exception('ProblemSolver: No feasible policies')

        return best_pol
