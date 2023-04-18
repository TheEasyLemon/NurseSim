import numpy as np

from ProblemInstance.ProblemInstance import ProblemInstance

class ProblemSolver:
    def __init__(self, prob: ProblemInstance) -> None:
        self.prob = prob

    def optimalColumn(self, shift: int) -> np.ndarray:
        pass