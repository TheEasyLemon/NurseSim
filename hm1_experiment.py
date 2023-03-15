'''
Experiments with the HM1 Algorithm.

Experiments:
1. Try to start from a blank slate (empty policy) and see if HM1 produces optimal for N_j = 1

Seems to work for N_j = 1, takes much longer to run though.

2. Try to solve with an iterative deepening approach (Solve for N_j = 1, then 2, then 3...etc.)

This proves that the theorem is correct, however it has a bad worst-case run time. It's possible
that we search through O(N_j 2^n), because we may end up doing an exhaustive search for every
n <= N_j. In practice however, the ID algorithm is faster than brute force search.

The concept of iterative deepening comes from chess and other minimax games with decision
trees. In order to determine which moves are good to a certain search depth, we start with
a decision tree with the minimum depth = 1, then perform alpha-beta pruning, then try with
depth = 2, prune, then depth = 3...etc. The key to this approach is that we repeat very
little work each time we "deepen" our search. This is because alpha-beta pruning is incredibly
efficient at eliminating possible moves.

In this case, we'd hope that we could eliminate some nurses or some combinations of nurses,
which might be very helpful. Let's see!

Dawson Ren
3/12/23
'''
from ProblemSolver.hm1 import HM1
from ProblemSolver.id import IterativeDeepening
from ProblemSolver.ProblemSolver import ProblemSolver
from utils.random_pi import generate_random_pi

def experiment_1():
    '''
    Identify a failure of HM1.
    '''
    m = 12
    max_iter = 1

    for i in range(max_iter):
        pi = generate_random_pi(m, m)
        ps = ProblemSolver(pi)
        ps_sol = ps.optimalPolicy()
        print('Optimal:\n', ps_sol)
        hm1_sol = HM1(pi, blank_slate=True)
        print('HM1:\n', hm1_sol)
        if pi.expectedRevenue(hm1_sol) < pi.expectedRevenue(ps_sol):
            print(f'Found inferior after {i} iterations:')
            print(pi)
            print('HM1 Revenue:', pi.expectedRevenue(hm1_sol))
            print('Optimal Revenue:', pi.expectedRevenue(ps_sol))
            return

    print(f'Unable to identify inferiority after {max_iter} iterations.')

def experiment_2():
    '''
    See if Iterative Deepening produces an optimal solution.

    Answer: Yes, ID produces an optimal solution, for 1000 replications on m = 10.
    '''
    m = 14
    max_iter = 1

    for i in range(max_iter):
        print('Iteration:', i + 1)
        pi = generate_random_pi(m, m)
        ps = ProblemSolver(pi)
        ps_sol = ps.optimalPolicy()
        # print('Optimal:\n', ps_sol)
        hm1_sol = IterativeDeepening(pi)
        print('HM1:\n', hm1_sol)
        if pi.expectedRevenue(hm1_sol) < pi.expectedRevenue(ps_sol):
            print(f'Found inferior after {i} iterations:')
            print(pi)
            print('Iterative Deepening Revenue:', pi.expectedRevenue(hm1_sol))
            print('Optimal Revenue:', pi.expectedRevenue(ps_sol))
            return


if __name__ == '__main__':
    # experiment_1()
    experiment_2()