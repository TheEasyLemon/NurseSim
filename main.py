'''
Driver code for finding the expected revenue for the revenue maximization problem.

Dawson Ren, 11/14/22
'''
import numpy as np

from ProblemInstance.ProblemInstance import ProblemInstance
from ProblemSolver.ProblemSolver import ProblemSolver

def main():
    m = n = 12
    P = np.random.rand(m, n)
    Q = np.random.rand(m, n)
    R = np.random.rand(m, n) * 10
    pi = ProblemInstance(P, Q, R)
    ps = ProblemSolver(pi)
    Y = ps.optimalPolicy()
    print(pi.expectedRevenue(Y))
    

if __name__ == '__main__':
    main()
