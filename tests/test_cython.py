'''
Simple speed comparison for Cython.
'''
import time

from utils.random_pi import generate_random_pi
from ProblemSolver.ProblemSolver import ProblemSolver

# 1. write make with the HH4 algorithm
# 2. see how long 20x1 takes

# we want to be able to run 200 replications of 20x1
# within a few minutes and compare to the 4-part HH1

def test_normal(N, size):
    start = time.perf_counter()

    for _ in range(N):
        pi = generate_random_pi(size, size)
        ps = ProblemSolver(pi)

        Y = ps.optimalColumn(0, cython=False)

    end = time.perf_counter()
    return end - start

def test_cython(N, size):
    start = time.perf_counter()

    for _ in range(N):
        pi = generate_random_pi(size, size)
        ps = ProblemSolver(pi)

        Y = ps.optimalColumn(0, cython=True)

    end = time.perf_counter()
    
    return end - start

if __name__ == '__main__':
    N = 200
    size = 10

    # 200 replications of 10x1 takes ~10 seconds.
    print(f'cython took {test_cython(N, size)} seconds.')