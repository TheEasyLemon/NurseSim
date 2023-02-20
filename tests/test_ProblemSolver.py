'''
Black-box unit testing for ProblemInstance.
'''
import numpy as np
import unittest
from ProblemSolver.ProblemSolver import ProblemSolver
from ProblemInstance.ProblemInstance import ProblemInstance
from ProblemReader.ProblemReader import loadProblemInstance

class TestProblemReader(unittest.TestCase):
    # def test1_data(self):
    #     pi = loadProblemInstance('tests/test1_data.txt')
    #     ps = ProblemSolver(pi)
    #     np.testing.assert_allclose(ps.bruteForceOptimalPolicy(), np.ones((3, 3)))
    #     np.testing.assert_allclose(ps.bruteForceOptimalPolicy(optimize=True), np.ones((3, 3)))

    # def test2_data(self):
    #     pi = loadProblemInstance('tests/test2_data.txt')
    #     ps = ProblemSolver(pi)
    #     sol = np.ones((3, 3))
    #     np.testing.assert_allclose(ps.bruteForceOptimalPolicy(), sol)
    #     np.testing.assert_allclose(ps.bruteForceOptimalPolicy(optimize=True), sol)

    # def test3_data(self):
    #     pi = loadProblemInstance('tests/test3_data.txt')
    #     ps = ProblemSolver(pi)
    #     np.testing.assert_allclose(ps.bruteForceOptimalPolicy(), np.ones((2, 2)))
    #     np.testing.assert_allclose(ps.bruteForceOptimalPolicy(optimize=True), np.ones((2, 2)))

    # def test_col_optimization(self):
    #     iters = 3
    #     m = n = 4
    #     P = np.random.rand(m, n)
    #     Q = np.random.rand(m, n)
    #     R = np.random.rand(m, n) * 10
    #     N = np.ones(shape=(n, ), dtype=np.int64)
    #     pi = ProblemInstance(P, Q, R, N)
    #     for _ in range(iters):
    #         ps = ProblemSolver(pi)
    #         np.testing.assert_allclose(ps.optimalPolicy(), ps.bruteForceOptimalPolicy())

    def test_dynamic_programming(self):
        iters = 100
        m = n = 5
        np.random.seed(0)
        P = np.random.rand(m, n).round(1)
        Q = np.random.rand(m, n).round(1)
        R = np.random.rand(m, n).round(1) * 10
        N = np.ones(shape=(n, ), dtype=np.int64)
        pi = ProblemInstance(P, Q, R, N)
        for _ in range(iters):
            ps = ProblemSolver(pi)
            op = ps.optimalPolicy()
            dp = ps.dynamicPolicy()
            # this condition doesn't always hold!
            # np.testing.assert_allclose(op, dp)
            np.testing.assert_allclose(pi.expectedRevenue(op), pi.expectedRevenue(dp))
    

if __name__ == '__main__':
    unittest.main()
