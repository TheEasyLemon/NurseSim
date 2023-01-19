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
    P = np.random.rand(m, n)
    P = np.array([[0.15, 0.90, 0.10, 0.45, 0.30],
                  [0.80, 0.05, 0.30, 0.60, 0.45],
                  [0.20, 0.90, 0.95, 0.10, 0.60],
                  [0.90, 0.90, 0.35, 0.30, 0.05],
                  [0.70, 0.60, 0.20, 0.75, 0.50]])
    Q = np.random.rand(m, n)
    Q = np.array([[0.60, 0.40, 0.30, 0.15, 0.15],
                  [0.80, 0.25, 0.65, 0.65, 0.65],
                  [0.60, 0.75, 0.20, 0.70, 0.05],
                  [0.15, 0.40, 0.80, 0.50, 0.70],
                  [0.95, 0.25, 0.15, 0.05, 0.75]])
    R = np.random.rand(m, n) * 10
    R = np.array([[0.5, 8.0, 1.5, 7.5, 7.5],
                  [9.5, 6.0, 6.5, 9.0, 8.0],
                  [8.5, 4.0, 7.0, 3.5, 3.0],
                  [2.0, 8.0, 5.5, 9.5, 6.5],
                  [4.0, 1.0, 2.5, 9.5, 1.5]])
    N = np.random.randint(1, int(n / 2), size=(n, ))
    N = np.array([3, 5, 4, 3, 2])
    pi = ProblemInstance(P, Q, R, N)
    ps = ProblemSolver(pi)
    Y = np.ones((m, n))
    monte_carlo_expected_revenue(pi, Y, alpha=0.1, e=0.005)
    

if __name__ == '__main__':
    main()
