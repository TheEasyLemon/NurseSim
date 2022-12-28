'''
Class containing relevant information for solving the revenue mangement problem.

expectedRevenue() -> float, returns expected revenue of the problem

Dawson Ren, 11/14/22
'''
import numpy as np
from typing import Union, Tuple, Generator
import itertools

class ProblemInstance:
    def __init__(self, P: np.ndarray, Q: np.ndarray, R: np.ndarray) -> None:
        self.P = P
        self.m, self.n = P.shape
        self.Q = Q
        self.R = self._expandRevenue(R)
        self.V = self.R * self.P * self.Q  # value matrix
        self.A = np.zeros((self.m, self.n)) # availability matrix

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
        key = (shift, self._serializeArray(y))
        if key in self.cache:
            return self.cache[key]
        else:
            return None

    def _calculateAvailability(self, Y: np.ndarray) -> np.ndarray:
        # the first row is simply the first row of Y
        self.A[0, :] = Y[0, :]

        for i in range(1, self.m):
            self.A[i, :] = 1 - Y[i - 1, :] * self.P[i - 1, :] * self.A[i - 1, :]

    def _calculateAvailabilityCol(self, shift: int, y: np.ndarray) -> np.ndarray:
        A_col = np.zeros((self.m,))
        A_col[0] = y[0]

        for i in range(1, self.m):
            A_col[i] = 1 - y[i - 1] * self.P[i - 1, shift] * A_col[i - 1]

        return A_col

    def expectedRevenueSlow(self, Y: np.ndarray) -> float:
        if Y.shape != (self.m, self.n):
            raise Exception('ProblemInstance: policy Y is of wrong shape')

        self._calculateAvailability(Y)
        return (self.V * Y * self.A).sum()

    def expectedRevenueCol(self, shift: int, y: np.ndarray) -> float:
        if y.shape[0] != self.m:
            raise Exception(f'ProblemInstance: policy column y is of wrong shape, got {y.shape}')

        expectedRevenue = 0

        lookup = self._lookup(shift, y)
        if lookup is None:
            expectedRevenueCol = (self.V[:, shift] * y * self._calculateAvailabilityCol(shift, y)).sum()
            expectedRevenue += expectedRevenueCol
            self._cache(shift, y, expectedRevenueCol)
        else:
            expectedRevenue += lookup

        return expectedRevenue
        

    def expectedRevenue(self, Y: np.ndarray) -> float:
        if Y.shape != (self.m, self.n):
            raise Exception('ProblemInstance: policy Y is of wrong shape')
        
        return sum([self.expectedRevenueCol(j, Y[:, j]) for j in range(self.n)])