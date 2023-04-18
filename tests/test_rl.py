import unittest
import numpy as np

from ProblemSolver.RL.rl import NurseRL

class TestProblemReader(unittest.TestCase):
    def test_setup_correct(self):
        rl = NurseRL(np.array([1, 1, 0]),
                     np.array([0.5, 0.3, 0.8]),
                     np.array([1.5, 1.6, 6.5]),
                     2)
        
        self.assertEqual(rl.m, 3)
        np.testing.assert_allclose(rl.v, np.array([0.75, 0.48, 5.2]))
        self.assertEqual(set(rl.S_plus), set([(1, 2), (2, 2), (3, 2), (4, 2),
                                         (2, 1), (3, 1), (4, 1), (4, 0)]))
        
        self.assertEqual(rl.step((1, 2), 0), ((2, 2), 0, False))
        self.assertEqual(rl.step((2, 2), 0), ((3, 2), 0, False))
        self.assertEqual(rl.step((3, 2), 0), ((4, 2), 0, True))
        self.assertEqual(rl.step((2, 1), 0), ((3, 1), 0, False))
        self.assertEqual(rl.step((3, 1), 0), ((4, 1), 0, True))

        self.assertEqual(rl.step((1, 2), 1), ((2, 1), 0.75, False))
        self.assertEqual(rl.step((2, 2), 1), ((3, 1), 0.48, False))
        self.assertEqual(rl.step((3, 2), 1), ((4, 2), 0, True))
        self.assertEqual(rl.step((2, 1), 1), ((4, 0), 0.48, True))
        self.assertEqual(rl.step((3, 1), 1), ((4, 1), 0, True))

    def test_policy_correct(self):
        # the optimal policy is [1 0 1]
        rl = NurseRL(np.array([0.6, 0.3, 0.6]),
                     np.array([0.3, 0.3, 0.8]),
                     np.array([1.5, 1.6, 6.5]),
                     2)
        print(rl.iteration())

if __name__ == '__main__':
    unittest.main()
        
