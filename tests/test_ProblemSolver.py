'''
Black-box unit testing for ProblemInstance.
'''
import numpy as np
import unittest
from ProblemSolver.ProblemSolver import ProblemSolver
from ProblemReader.ProblemReader import loadProblemInstance

class TestProblemReader(unittest.TestCase):
    def test1_data(self):
        pi = loadProblemInstance('tests/test1_data.txt')
        ps = ProblemSolver(pi)
        np.testing.assert_allclose(ps.bruteForceOptimalPolicy(), np.zeros((3, 3)))

    def test2_data(self):
        pi = loadProblemInstance('tests/test2_data.txt')
        ps = ProblemSolver(pi)
        np.testing.assert_allclose(ps.bruteForceOptimalPolicy(), np.zeros((3, 3)))

    def test3_data(self):
        pi = loadProblemInstance('tests/test3_data.txt')
        ps = ProblemSolver(pi)
        np.testing.assert_allclose(ps.bruteForceOptimalPolicy(), np.zeros((3, 3)))

if __name__ == '__main__':
    unittest.main()
