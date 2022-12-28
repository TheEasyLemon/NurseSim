'''
Black-box unit testing for ProblemInstance.
'''
import numpy as np
import unittest
from ProblemSolver.ProblemSolver import ProblemSolver
from ProblemInstance.ProblemInstance import ProblemInstance
from ProblemReader.ProblemReader import loadProblemInstance

class TestProblemReader(unittest.TestCase):
    def test1_data(self):
        pi = loadProblemInstance('tests/test1_data.txt')
        ps = ProblemSolver(pi)
        np.testing.assert_allclose(ps.bruteForceOptimalPolicy(), np.ones((3, 3)))
        np.testing.assert_allclose(ps.bruteForceOptimalPolicy(optimize=True), np.ones((3, 3)))

    def test2_data(self):
        pi = loadProblemInstance('tests/test2_data.txt')
        ps = ProblemSolver(pi)
        sol = np.ones((3, 3))
        sol[0, 2] = 0
        np.testing.assert_allclose(ps.bruteForceOptimalPolicy(), sol)
        np.testing.assert_allclose(ps.bruteForceOptimalPolicy(optimize=True), sol)

    def test3_data(self):
        pi = loadProblemInstance('tests/test3_data.txt')
        ps = ProblemSolver(pi)
        np.testing.assert_allclose(ps.bruteForceOptimalPolicy(), np.ones((2, 2)))
        np.testing.assert_allclose(ps.bruteForceOptimalPolicy(optimize=True), np.ones((2, 2)))

    def test_iters(self):
        iters = 1
        m = 4
        n = 4
        P = np.random.rand(m, n)
        Q = np.random.rand(m, n)
        R = np.random.rand(m, n) * 10
        pi = ProblemInstance(P, Q, R)

        for _ in range(iters):
            Y = np.random.randint(0, 2, size=(m, n))
            ps = ProblemSolver(pi)
            np.testing.assert_allclose(ps.optimalPolicy(), ps.bruteForceOptimalPolicy())

if __name__ == '__main__':
    unittest.main()
