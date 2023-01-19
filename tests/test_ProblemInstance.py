'''
Black-box unit testing for ProblemInstance.
'''
import numpy as np
import unittest
from ProblemReader.ProblemReader import loadProblemInstance
from ProblemInstance.ProblemInstance import ProblemInstance

class TestProblemReader(unittest.TestCase):
    def test1_data(self):
        pi = loadProblemInstance('tests/test1_data.txt')
        # calculated by hand
        self.assertAlmostEqual(pi.expectedRevenue(np.array([[1, 0, 0],
                                                            [0, 1, 0],
                                                            [0, 0, 1]])),
                                                            4.79)

    def test2_data(self):
        pi = loadProblemInstance('tests/test2_data.txt')
        # calculated by hand
        self.assertAlmostEqual(pi.expectedRevenue(np.array([[1, 1, 0],
                                                            [1, 1, 1],
                                                            [0, 0, 1]])),
                                                            10.2185)

    def test3_data(self):
        pi = loadProblemInstance('tests/test3_data.txt')
        # calculated by hand
        self.assertAlmostEqual(pi.expectedRevenue(np.array([[1, 1],
                                                            [1, 1]])),
                                                            6.59)
            

if __name__ == '__main__':
    unittest.main()
