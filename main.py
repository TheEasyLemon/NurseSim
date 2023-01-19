'''
Driver code for finding the expected revenue for the revenue maximization problem.

Dawson Ren, 11/14/22
'''
import numpy as np

from ProblemInstance.ProblemInstance import ProblemInstance
from ProblemSolver.ProblemSolver import ProblemSolver
from utils.monte_carlo import monte_carlo_expected_revenue

def main():
    m = n = 5
    np.random.seed = 1
    # P = np.random.rand(m, n)
    P = np.array([[0.15894727, 0.92348405, 0.12173334, 0.43194288, 0.29603429],
                  [0.79605674, 0.05844282, 0.29542847, 0.57236066, 0.45246991],
                  [0.20760343, 0.89177188, 0.97770233, 0.08823165, 0.61236282],
                  [0.90540417, 0.91179464, 0.36090483, 0.27709284, 0.00475731],
                  [0.70341418, 0.59510609, 0.19864719, 0.76706991, 0.47014319]])
    # Q = np.random.rand(m, n)
    Q = np.array([[0.59515262, 0.39166604, 0.31603607, 0.15765413, 0.14296166],
                  [0.78420092, 0.25128868, 0.67728037, 0.63178488, 0.63643852],
                  [0.57762801, 0.72133602, 0.19615321, 0.71388181, 0.03690623],
                  [0.13011557, 0.38737188, 0.81872936, 0.53172107, 0.68577018],
                  [0.93466361, 0.22158319, 0.14127877, 0.03933603, 0.74952099]])
    # R = np.random.rand(m, n) * 10
    R = np.array([[0.43609019, 7.94216841, 0.13707617, 7.45920265, 7.657821  ],
                  [9.61267686, 5.90726674, 6.72978904, 9.17287845, 7.93369424],
                  [8.65299869, 4.06612351, 7.17595568, 3.57753835, 3.24503143],
                  [2.13921865, 7.81735176, 5.85126927, 9.99631899, 0.35283344],
                  [3.83235046, 0.87365333, 2.23326097, 9.9783341,  1.64480582]])
    # N = np.ones((20, 1))
    N = np.array([1, 1, 1, 1, 1])
    pi = ProblemInstance(P, Q, R, N)
    ps = ProblemSolver(pi)
    Y = np.ones((5, 5))
    monte_carlo_expected_revenue(pi, Y)
    

if __name__ == '__main__':
    main()
