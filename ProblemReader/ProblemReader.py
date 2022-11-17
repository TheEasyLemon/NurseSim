'''
Loads in the data, creating a ProblemInstance.

loadProblemInstance(path: str) -> ProblemInstance

Dawson Ren, 11/14/22
'''
from typing import List
import numpy as np

from ProblemInstance.ProblemInstance import ProblemInstance as PI

def parseLine(line: str, path: str, i: int) -> List[float]:
    try:
        return [float(num) for num in line.split(',')]
    except:
        raise Exception(f'Failed to parse a value in file: {path}, line {i}')

def loadProblemInstance(path: str) -> PI:
    # number of nurses and number of shifts
    m = n = None
    P = []
    Q = []
    r = []
    Y = []

    with open(path, 'r') as file:
        i = 1
        while (line := file.readline().strip()) != '':
            line_nums = parseLine(line, path, i)
            if m is None:
                m = len(line_nums)
            elif m != len(line_nums):
                raise Exception(f'Uneven line length found while loading problem in file: {path}, line {i}')

            P.append(line_nums)
            i += 1

        n = len(P)
        i += 1

        while (line := file.readline().strip()) != '':
            line_nums = parseLine(line, path, i)
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

            r = line_nums
            i += 1

        i += 1

        while (line := file.readline().strip()) != '':
            line_nums = parseLine(line, path, i)
            if m != len(line_nums):
                raise Exception(f'Uneven line length found while loading problem in file: {path}, line {i}')

            if not all(map(lambda n: n in [0, 1], line_nums)):
                raise Exception(f'Only binary values allowed in Y matrix, file: {path}, line {i}')

            Y.append(line_nums)
            i += 1

        if len(Y) != n:
            raise Exception(f'Dimensions of P and Y don\'t match in file: {path}, line {i}')

    pi = PI(np.array(P, dtype=np.float64), np.array(Q, dtype=np.float64), np.array(r, dtype=np.float64), np.array(Y, dtype=np.int8))
    return pi
