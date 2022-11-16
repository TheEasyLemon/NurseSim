'''
Class containing relevant information for solving the revenue mangement problem.

expectedRevenue() -> float, returns expected revenue of the problem

Dawson Ren, 11/14/22
'''
import numpy as np

class ProblemInstance:
    def __init__(self, P: np.ndarray, Q: np.ndarray, r: np.ndarray, Y: np.ndarray) -> None:
        self.P = P
        self.Q = Q
        self.r = r
        self.Y = Y

    def expectedRevenue(self) -> float:
        pass