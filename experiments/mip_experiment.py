'''
Experiment with the MIP solver in ProblemSolver/mip.py.
'''
import sys
from ProblemSolver.mip import MIP
from ProblemSolver.ProblemSolver import ProblemSolver
from utils.random_pi import generate_random_pi

def test_batch(rounds = 1000):
    correct = 0
    rev_total = 0
    rev_captured = 0

    for i in range(rounds):
        print(i)
        pi = generate_random_pi(4, 1, ones=True, constant_revenue_per_shift=True, round=2)
        mip_soln = MIP(pi)
        opt_soln = ProblemSolver(pi).optimalPolicy()
        mip_rev = pi.expectedRevenueCol(0, mip_soln[:, 0]) 
        opt_rev = pi.expectedRevenueCol(0, opt_soln[:, 0])

        if abs(mip_rev - opt_rev) < sys.float_info.epsilon:
            correct += 1

        rev_total += opt_rev
        rev_captured += mip_rev
        
        # print(mip_soln)
        # print(opt_soln)
        # print(pi.expectedRevenueCol(0, mip_soln[:, 0]))
        # print(pi.expectedRevenueCol(0, opt_soln[:, 0]))

    print('Revenue captured:', rev_captured / rev_total)
    print('Found optimal solution at this percentage:', correct / rounds)

if __name__ == '__main__':
    test_batch(100)