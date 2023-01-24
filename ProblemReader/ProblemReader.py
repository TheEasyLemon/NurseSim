'''
Loads in the data, creating a ProblemInstance.

loadProblemInstance(path: str) -> ProblemInstance

Dawson Ren, 11/14/22
'''
from typing import List
import numpy as np

from ProblemInstance.ProblemInstance import ProblemInstance as PI

def parseLine(line: str, path: str, i: int, prob: bool = False) -> List[float]:
    try:
        floats = [float(num) for num in line.split(',')]
        if prob:
            for f in floats:
                if f > 1: raise Exception(f'Not possible to have probability greater than 1 in file: {path}, line {i}')
        return floats
    except ValueError:
        raise Exception(f'Failed to parse a value in file: {path}, line {i}')

def loadProblemInstance(path: str) -> PI:
    # number of nurses and number of shifts
    m = n = None
    P = []
    Q = []
    R = []
    N = []

    with open(path, 'r') as file:
        i = 1
        while (line := file.readline().strip()) != '':
            line_nums = parseLine(line, path, i, True)
            if m is None:
                m = len(line_nums)
            elif m != len(line_nums):
                raise Exception(f'Uneven line length found while loading problem in file: {path}, line {i}')

            P.append(line_nums)
            i += 1

        n = len(P)
        i += 1

        while (line := file.readline().strip()) != '':
            line_nums = parseLine(line, path, i, True)
            if m != len(line_nums):
                raise Exception(f'Uneven line length found while loading problem in file: {path}, line {i}')

            Q.append(line_nums)
            i += 1

        if len(Q) != n:
            raise Exception(f'Dimensions of P and Q don\'t match in file: {path}, line {i}')
        i += 1

        while (line := file.readline().strip()) != '':
            line_nums = parseLine(line, path, i)
            if m != len(line_nums):
                raise Exception(f'Uneven line length found while loading problem in file: {path}, line {i}')

            R.append(line_nums)
            i += 1

        if len(R) not in [1, n]:
            raise Exception(f'Dimensions of P and R don\'t match in file: {path}, line {i}')

        i += 1

        while (line := file.readline().strip()) != '':
            line_nums = parseLine(line, path, i)
            if m != len(line_nums):
                raise Exception(f'Uneven line length found while loading problem in file: {path}, line {i}')

            N.append(line_nums)
            i += 1

        if len(N) != 1:
            raise Exception(f'Number of nurses per shift should only specify per shift: {path}, line {i}')

    pi = PI(np.array(P, dtype=np.float64), np.array(Q, dtype=np.float64), np.array(R, dtype=np.float64), np.array(N, dtype=np.int64).flatten())
    return pi
