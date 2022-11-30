'''
Class containing relevant information for solving the revenue mangement problem.

expectedRevenue() -> float, returns expected revenue of the problem

Dawson Ren, 11/14/22
'''
import numpy as np
from typing import Union, Tuple, Generator
import itertools

class ProblemInstance:
    def __init__(self, P: np.ndarray, Q: np.ndarray, r: np.ndarray) -> None:
        self.P = P
        self.m, self.n = P.shape
        self.Q = Q
        self.R = self._expandRevenue(r)

        self.cache = dict()

    def _expandRevenue(self, r: np.ndarray) -> np.ndarray:
        if r.shape == self.P.shape:
            return r
        
        R = np.tile(r, (self.n, 1))
        return R

    def _serializeArray(self, A: np.ndarray) -> bytes:
        # convert to bytes
        return A.tobytes()

    def _deserializeArray(self, b: bytes, dtype) -> np.ndarray:
        # reshape from bytes to m x n matrix
        return np.reshape(np.frombuffer(b, dtype=dtype), (self.m, self.n))

    def _cache(self, nurse: int, Z: np.ndarray, value: float) -> None:
        # store in cache as nurse, availability matrix tuple mapping to expected value
        self.cache[(nurse, self._serializeArray(Z))] = value

    def _lookup(self, nurse: int, Z: np.ndarray) -> Union[float, None]:
        # lookup from cache
        key = (nurse, self._serializeArray(Z))
        if key in self.cache:
            return self.cache[key]
        else:
            return None

    def _getPossibleAssignments(self, nurse: int, Z: np.ndarray) -> Generator[Tuple[int, ...], None, None]:
        # get all possible assignments for a nurse.
        # -1 means that the nurse decides not to schedule that shift.
        # 0 means the shift was unavailable.
        # 1 means the nurse took the shift.
        row = Z[nurse, :]
        ret_val = row.copy()
        ones = np.where(row == 1)[0]
        for replace in itertools.product([-1, 1], repeat=len(ones)):
            ret_val[ones] = replace
            yield tuple(ret_val)
        return

    def _getAssignmentProbability(self, nurse: int, assignment: Tuple[int, ...]) -> float:
        # get probability of the shift combination, conditioning on the shifts with 0 being unavailable
        prob = 1
        for j, assigned in enumerate(assignment):
            if assigned == -1:
                prob *= 1 - self.P[nurse, j]
            elif assigned == 1:
                prob *= self.P[nurse, j]
        return prob

    def _eliminateShifts(self, c: Tuple[int,...], Z: np.ndarray) -> None:
        # make shifts unavailable. Having a 0 in the availability matrix means the shift is unavailable.
        Z_new = Z.copy()
        Z_new[:, np.arange(0, len(c))[np.array(c) == 1]] = 0
        return Z_new

    def _expectedRevenueLoop(self, nurse: int, Z: np.ndarray) -> float:
        # base case
        if nurse >= self.m:
            return 0

        print(f'Starting for nurse {nurse} and with availability matrix\n{Z}')

        # lookup from cache
        lookup = self._lookup(nurse, Z)
        if lookup is not None:
            print(f'Lookup found value of {lookup}, returning.')
            return lookup

        # recursive case
        expected_rev = 0

        for c in self._getPossibleAssignments(nurse, Z):
            print(f'With assignment {c} for nurse {nurse}, ', end='')
            p = self._getAssignmentProbability(nurse, c)
            print(f'Got probability {p}, ', end='')
            # immediate expected revenue
            imm_exp_rev = sum([self.Q[nurse, j] * self.R[nurse, j] * max(cj, 0) for j, cj in enumerate(c)])
            print(f'Got immediate return {imm_exp_rev}.')
            # future expected revenue for all nurses afterwards
            print(f'Recursing downwards to next nurse.')
            fut_exp_rev = self._expectedRevenueLoop(nurse + 1, self._eliminateShifts(c, Z))
            print(f'Got future return {fut_exp_rev}.')
            expected_rev += p * (imm_exp_rev + fut_exp_rev)

        self._cache(nurse, Z, expected_rev)

        return expected_rev

    def _expectedRevenueOld(self, Y: np.ndarray) -> float:
        Z = Y.copy()

        print('\nStarting calculations...\n')
        print(f'Using matrix P = \n{self.P}')
        print(f'Using matrix Q = \n{self.Q}')
        print(f'Using vector r = \n{self.R}')
        print(f'Using matrix Y = \n{Y}')

        return self._expectedRevenueLoop(0, Z)

    def _calculateAvailability(self, Y: np.ndarray) -> np.ndarray:
        A = np.zeros((self.m, self.n))

        # the first row is simply the first row of Y
        A[0, :] = Y[0, :]

        for i in range(1, self.m):
            A[i, :] = 1 - Y[i - 1, :] * self.P[i - 1, :] * A[i - 1, :]

        return A

    def expectedRevenue(self, Y: np.ndarray) -> float:
        if Y.shape != (self.m, self.n):
            raise Exception('ProblemInstance: policy Y is of wrong shape')

        A = self._calculateAvailability(Y)
        return (self.R * self.P * self.Q * Y * A).sum()