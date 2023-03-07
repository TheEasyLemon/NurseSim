import numpy as np
import matplotlib.pyplot as plt

from ProblemSolver.hm1 import HM1
from ProblemSolver.ProblemSolver import ProblemSolver
from utils.random_pi import generate_random_pi

def visualize_error_with_k():
    # m nurses, just one schedule
    m = 15
    n = 1
    k_vals = np.arange(1, 5)
    errs = np.zeros(k_vals.size)

    for i, k in enumerate(k_vals):
        pi = generate_random_pi(m, n)
        ps = ProblemSolver(pi)
        hm1_policy = HM1(pi, int(k))
        opt_policy = ps.optimalPolicyHeuristic()
        rev_opt = pi.expectedRevenue(opt_policy)
        rev_hm1 = pi.expectedRevenue(hm1_policy)
        errs[i] = 100 * (rev_hm1 - rev_opt) / rev_opt

    print(errs)

    plt.plot(k_vals, errs)
    plt.title('Relative Error of HM1 policy with Various K')
    plt.xlabel('K value')
    plt.ylabel('Relative Error (%)')
    plt.show()

if __name__ == '__main__':
    visualize_error_with_k()