'''
Black-box unit testing for ProblemInstance.
'''
import numpy as np
import unittest
from ProblemReader.ProblemReader import loadProblemInstance
from ProblemInstance.ProblemInstance import ProblemInstance
from utils.monte_carlo import monte_carlo_expected_revenue
from utils.random_pi import generate_random_pi

class TestProblemInstance(unittest.TestCase):
    def test_random(self):
        num = 10
        m = 7

        for _ in range(num):
            pi = generate_random_pi(m, m)
            print(pi)
            Y = np.random.randint(0, 2, size=(m, m), dtype=np.int64)
            print(Y)
            print(pi.calculateAvailability(Y))
            exact = pi.expectedRevenue(Y)
            mc = monte_carlo_expected_revenue(pi, Y)
            print(exact - mc)
            self.assertAlmostEqual(exact, mc, delta=0.1)
    
    def test1_data(self):
        pi = loadProblemInstance('tests/test1_data.txt')
        Y = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        self.assertAlmostEqual(pi.expectedRevenue(Y), monte_carlo_expected_revenue(pi, Y), delta=0.1)

    def test2_data(self):
        pi = loadProblemInstance('tests/test2_data.txt')
        Y = np.array([[1, 1, 0],
                      [1, 1, 1],
                      [0, 0, 1]])
        self.assertAlmostEqual(pi.expectedRevenue(Y), monte_carlo_expected_revenue(pi, Y), delta=0.1)

    def test3_data(self):
        pi = loadProblemInstance('tests/test3_data.txt')
        # calculated by hand
        Y = np.array([[1, 1],
                      [1, 1]])
        self.assertAlmostEqual(pi.expectedRevenue(Y), monte_carlo_expected_revenue(pi, Y), delta=0.1)

    def test4_data(self):
        pi = loadProblemInstance('tests/test4_data.txt')
        # calculated by hand
        Y = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]])
        self.assertAlmostEqual(pi.expectedRevenue(Y), monte_carlo_expected_revenue(pi, Y), delta=0.1)

if __name__ == '__main__':
    unittest.main()
