'''
Class containing relevant information for solving the revenue mangement problem.

expectedRevenue() -> float, returns expected revenue of the problem

Dawson Ren, 11/14/22
'''
from typing import Union
from itertools import combinations

import numpy as np
import pyximport # for Cython interop
pyximport.install()
import utils.find_availability as fa

class ProblemInstance:
    def __init__(self, P: np.ndarray, Q: np.ndarray, R: np.ndarray, N: np.ndarray, cython=False) -> None:
        self.P = P
        self.m, self.n = P.shape
        self.Q = Q
        self.R = self._expandRevenue(R)
        self.N = N
        self.V = self.R * self.P * self.Q  # value matrix

        self.cache = dict()
        self.use_cython = cython

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
    
    def _calculateAvailabilityCol(self, shift: int, y: np.ndarray) -> np.ndarray:
        '''
        Calculate availability of a column.
        '''
        if self.use_cython: return fa.calculateAvailabilityCol(self.N[shift], self.m, y, self.P[:, shift])
        
        N = self.N[shift]
        A = np.triu(np.ones((y.size, N)), 0) # the row gives the nurse, the column gives the availability when j shifts allowed
        Py = self.P[:, shift] * y.ravel() # the elementwise product of the col of P corresponding to this shift and y

        for i in range(1, self.m):
            A[i, 0] = (1 - Py[i - 1]) * A[i - 1, 0]

        for n in range(1, N):
            for i in range(n + 1, self.m):
                for tup in combinations(np.arange(i)[Py[:i] != 0], n):
                    prob = 1
                    for k in range(i):
                        if k in tup:
                            prob *= Py[k]
                        else:
                            prob *= (1 - Py[k])
                    A[i, n] += prob

        A_col = A.sum(axis=1)
        A_col[:N] = 1 # first N nurses get availability 1
        return A_col
    
    def calculateAvailability(self, Y: np.ndarray) -> np.ndarray:
        if Y.shape != (self.m, self.n):
            raise Exception('ProblemInstance: policy Y is of wrong shape')
        
        return np.hstack([np.array(self._calculateAvailabilityCol(j, Y[:, j])).reshape((self.m, 1)) for j in range(self.n)])

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
        
    def expectedRevenueColSpecifyN(self, shift: int, y: np.ndarray, n: int):
        if y.shape[0] != self.m:
            raise Exception(f'ProblemInstance: policy column y is of wrong shape, got {y.shape}')
        
        return (self.V[:, shift] * y * fa.calculateAvailabilityCol(n, self.m, y, self.P[:, shift])).sum()

    def expectedRevenue(self, Y: np.ndarray) -> float:
        if Y.shape != (self.m, self.n):
            raise Exception('ProblemInstance: policy Y is of wrong shape')
        
        return sum([self.expectedRevenueCol(j, Y[:, j]) for j in range(self.n)])
    
    def copy(self):
        new_pi = ProblemInstance(self.P.copy(), self.Q.copy(), self.R.copy(), self.N.copy())
        return new_pi
    
    def __str__(self) -> str:
        return f'P:\n{self.P}\n\nV:\n{self.V}\n\nN: {self.N}'