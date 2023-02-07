'''
Class containing relevant information for solving the revenue mangement problem.

expectedRevenue() -> float, returns expected revenue of the problem

Dawson Ren, 11/14/22
'''
import numpy as np
from typing import Union
from itertools import combinations

class ProblemInstance:
    def __init__(self, P: np.ndarray, Q: np.ndarray, R: np.ndarray, N: np.ndarray) -> None:
        self.P = P
        self.m, self.n = P.shape
        self.Q = Q
        self.R = self._expandRevenue(R)
        self.N = N
        self.V = self.R * self.P * self.Q  # value matrix

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

    def _calculateAvailabilityCol(self, shift: int, y: np.ndarray) -> np.ndarray:
        N = self.N[shift]
        A = np.triu(np.ones((y.size, N)), 0) # the row gives the nurse, the column gives the availability when j shifts allowed
        Py = self.P[:, shift] * y.ravel() # the elementwise product of the col of P corresponding to this shift and y
        
        # let infs happen when we divide by 0
        with np.errstate(divide='ignore'):
            flip = Py / (1 - Py) # given a state where index i is not scheduled, multiply by this to switch to a state
                                 # where i is scheduled

        # first fill in 0th column of A
        for i in range(1, self.m):
            A[i, 0] = (1 - Py[i - 1]) * A[i - 1, 0]

        for n in range(1, N):
            for i in range(n + 1, self.m): # for every nurse, starting after the entries guaranted to be 1
                flips = 0 # keeps track of flips
                for tup in combinations(range(i), n):
                    prod = 1
                    for t in tup:
                        prod *= flip[t]
                    flips += prod
                A[i, n] = A[i, 0] * flips

        A_col = A.sum(axis=1)
        A_col[:N] = 1 # first N nurses get availability 1
       
        return A_col
    
    def calculateAvailability(self, Y: np.ndarray) -> np.ndarray:
        if Y.shape != (self.m, self.n):
            raise Exception('ProblemInstance: policy Y is of wrong shape')
        
        return np.hstack([self._calculateAvailabilityCol(j, Y[:, j]).reshape((self.m, 1)) for j in range(self.n)])

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

    def expectedRevenue(self, Y: np.ndarray) -> float:
        if Y.shape != (self.m, self.n):
            raise Exception('ProblemInstance: policy Y is of wrong shape')
        
        return sum([self.expectedRevenueCol(j, Y[:, j]) for j in range(self.n)])
    
    def copy(self):
        new_pi = ProblemInstance(self.P, self.Q, self.R, self.N)
        return new_pi