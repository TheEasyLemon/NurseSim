'''
Black-box unit testing for ProblemInstance.
'''
import numpy as np
import unittest
from ProblemReader.ProblemReader import loadProblemInstance
from ProblemInstance.ProblemInstance import ProblemInstance
from utils.monte_carlo import monte_carlo_expected_revenue

class TestProblemInstance(unittest.TestCase):
    def test1_data(self):
        pi = loadProblemInstance('tests/test1_data.txt')
        Y = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        self.assertAlmostEqual(pi.expectedRevenue(Y), monte_carlo_expected_revenue(pi, Y), places=1)

    def test2_data(self):
        pi = loadProblemInstance('tests/test2_data.txt')
        Y = np.array([[1, 1, 0],
                      [1, 1, 1],
                      [0, 0, 1]])
        self.assertAlmostEqual(pi.expectedRevenue(Y), monte_carlo_expected_revenue(pi, Y), places=1)

    def test3_data(self):
        pi = loadProblemInstance('tests/test3_data.txt')
        # calculated by hand
        Y = np.array([[1, 1],
                      [1, 1]])
        self.assertAlmostEqual(pi.expectedRevenue(Y), monte_carlo_expected_revenue(pi, Y), places=1)
            

if __name__ == '__main__':
    unittest.main()
