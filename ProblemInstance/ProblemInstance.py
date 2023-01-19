'''
Class containing relevant information for solving the revenue mangement problem.

expectedRevenue() -> float, returns expected revenue of the problem

Dawson Ren, 11/14/22
'''
import numpy as np
from typing import Union, Tuple, Generator
import itertools

class ProblemInstance:
    def __init__(self, P: np.ndarray, Q: np.ndarray, R: np.ndarray, N: np.ndarray) -> None:
        self.P = P
        self.m, self.n = P.shape
        self.Q = Q
        self.R = self._expandRevenue(R)
        self.N = N
        self.V = self.R * self.P * self.Q  # value matrix

        # values that will be precalculated later during the expected revenue step
        self.PY = np.zeros((self.m, self.n)) # the elementwise product of P and Y (the policy matrix)
        self.A0 = np.zeros((self.m, self.n)) # the probability that no nurses are scheduled
        self.A1 = np.zeros((self.m, self.n)) # the probability that one nurse is scheduled

        self.cache = dict()

    def _expandRevenue(self, r: np.ndarray) -> np.ndarray:
        if r.shape == self.P.shape:
            return r
        
        R = np.tile(r, (self.m, 1))
        return R

    def _serializeArray(self, A: np.ndarray) -> bytes:
        # convert to bytes
        return A.tobytes()

    def _deserializeArray(self, b: bytes, dtype) -> np.ndarray:
        # reshape from bytes to m x n matrix
        return np.reshape(np.frombuffer(b, dtype=dtype), (self.m, self.n))

    def _cache(self, shift: int, y: np.ndarray, value: float) -> None:
        # store in cache as nurse, availability matrix tuple mapping to expected value
        self.cache[(shift, self._serializeArray(y))] = value

    def _lookup(self, shift: int, y: np.ndarray) -> Union[float, None]:
        # lookup from cache
        # SOMETHING IS GOING WRONG HERE! Non-unique keys?
        key = (shift, self._serializeArray(y))
        # return None
        if key in self.cache:
            return self.cache[key]
        else:
            return None

    def _pre_calculate(self, Y: np.ndarray) -> None:
        self.PY = self.P * Y
        one_minus_PY = 1 - self.PY
        self.A0[0, :] = 1 # shifts are always available for the first nurse
        self.A1[[0, 1], :] = 1 # shifts are always available for the first and second

        # for every nurse, populate A0
        for i in range(1, self.m):
            self.A0[i, :] = self.A0[i - 1, :] * one_minus_PY[i - 1, :]

        # for every nurse, populate A1
        for i in range(2, self.m):
            # for every nurse
            for j in range(self.n):
                prob = 0
                for k in range(0, i - 1):
                    mult_prob = 1
                    for l in range(0, i - 1):
                        if l != k:
                            mult_prob *= 1 - self.P[l, j] * Y[l, j]

                    prob += mult_prob * self.P[k, j] * Y[k, j]

                self.A1[i, j] = prob

    def _calculateAvailabilityCol(self, shift: int, y: np.ndarray) -> np.ndarray:
        A_col = np.zeros((self.m,))
        A_col[0] = y[0]

        for i in range(1, self.m):
            A_col[i] = (1 - y[i - 1] * self.P[i - 1, shift]) * A_col[i - 1]

        return A_col

    def expectedRevenueCol(self, shift: int, y: np.ndarray) -> float:
        if y.shape[0] != self.m:
            raise Exception(f'ProblemInstance: policy column y is of wrong shape, got {y.shape}')

        lookup = self._lookup(shift, y)
        if lookup is None:
            expectedRevenueCol = (self.V[:, shift] * y * self._calculateAvailabilityCol(shift, y)).sum()
            self._cache(shift, y, expectedRevenueCol)
            return expectedRevenueCol
        else:
            return lookup

    def expectedRevenue(self, Y: np.ndarray, pre_calc=False) -> float:
        if Y.shape != (self.m, self.n):
            raise Exception('ProblemInstance: policy Y is of wrong shape')

        if pre_calc:
            self._pre_calculate(Y)
            # print(self.A1)
            # print(Y)
            # print(self.P)
            return (self.V * Y * (self.A0 + self.A1)).sum()
        
        return sum([self.expectedRevenueCol(j, Y[:, j]) for j in range(self.n)])