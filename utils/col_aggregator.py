'''
Common operation that we need - aggregating the columns
of individual solutions for different shifts.
'''
from typing import Callable

import numpy as np

from ProblemInstance.ProblemInstance import ProblemInstance

def col_aggregator(col_func: Callable[[ProblemInstance, int], np.ndarray]):
    def optimal(pi: ProblemInstance):
        return np.hstack([col_func(pi, i) for i in range(pi.n)])
    return optimal