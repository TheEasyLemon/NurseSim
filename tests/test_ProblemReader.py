'''
Unit testing for ProblemReader.
'''
import numpy as np
import unittest
from ProblemReader.ProblemReader import loadProblemInstance

class TestProblemReader(unittest.TestCase):
    def test1_data(self):
        pi = loadProblemInstance('tests/test1_data.txt')

        np.testing.assert_allclose(pi.P, np.array([[0.5, 0.25, 0.5],
                                                   [0.75, 0.5, 0.9],
                                                   [0.7, 0.5, 0.2]]))
        np.testing.assert_allclose(pi.Q, np.array([[0.9, 0.9, 0.8],
                                                   [0.8, 0.5, 0.75],
                                                   [0.5, 0.3, 0.7]]))
        np.testing.assert_allclose(pi.R, np.array([[4, 10, 3.5],
                                                   [4, 10, 3.5],
                                                   [4, 10, 3.5]]))
        np.testing.assert_allclose(pi.N, np.array([[1, 1, 1]]))

    def test2_data(self):
        pi = loadProblemInstance('tests/test2_data.txt')

        np.testing.assert_allclose(pi.P, np.array([[0.5, 0.25, 0.5],
                                                   [0.75, 0.5, 0.9],
                                                   [0.7, 0.5, 0.2]]))
        np.testing.assert_allclose(pi.Q, np.array([[0.9, 0.9, 0.8],
                                                   [0.8, 0.5, 0.75],
                                                   [0.5, 0.3, 0.7]]))
        np.testing.assert_allclose(pi.R, np.array([[4, 10, 3.5],
                                                   [5, 3, 6],
                                                   [7, 2, 4]]))
        np.testing.assert_allclose(pi.N, np.array([[2, 3, 2]]))

if __name__ == '__main__':
    unittest.main()
