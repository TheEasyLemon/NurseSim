'''
Black-box unit testing for ProblemInstance.
'''
import numpy as np
import unittest
from ProblemReader.ProblemReader import loadProblemInstance

class TestProblemReader(unittest.TestCase):
    def test1_data(self):
        pi = loadProblemInstance('tests/test1_data.txt')
        # calculated by hand
        self.assertAlmostEqual(pi.expectedRevenue(), 9.5365)

    def test2_data(self):
        pi = loadProblemInstance('tests/test2_data.txt')
        # calculated by hand
        # self.assertAlmostEqual(pi.expectedRevenue(), 4.79)

    def test3_data(self):
        pi = loadProblemInstance('tests/test3_data.txt')
        # calculated by hand
        self.assertAlmostEqual(pi.expectedRevenue(), 6.59)

if __name__ == '__main__':
    unittest.main()
