'''
Mixed Integer Programming formulation.
'''
from itertools import combinations
import warnings

# library imports
import numpy as np
from sympy import *
from sympy.matrices.dense import matrix_multiply_elementwise

# pulp (Python LP solver, interfaces with CPLEX/Gurobi/other solvers)
from pulp import *

from ProblemInstance.ProblemInstance import ProblemInstance
from ProblemSolver.ProblemSolver import ProblemSolver
from utils.col_aggregator import col_aggregator


def calculate_symbolic_availability(p, y, N, m):
    m = len(p)
    a = np.triu(np.ones((m, N), dtype=object), 0)

    for i in range(1, m):
        a[i, 0] = (1 - p[i - 1] * y[i - 1]) * a[i - 1, 0]

    # for N_j = 2 and above
    for n in range(1, N):
        for i in range(n + 1, m):
            for tup in combinations(range(i), n):
                prob = 1
                for k in range(i):
                    if k in tup:
                        prob *= p[k] * y[k]
                    else:
                        prob *= (1 - p[k] * y[k])
                a[i, n] += prob

    a = a.sum(axis=1)

    # enforce first N nurses get availability 1
    for i in range(N):
        a[i] = 1
        
    return Matrix(a)

def get_revenue_function(m, N):
    # r - the revenue of the shift
    r = symbols('r')
    q = Matrix(symbols(f'q1:{m+1}'))
    y = Matrix(symbols(f'Y1:{m+1}'))
    p = Matrix(symbols(f'p1:{m+1}'))

    # now, we calculate availability
    a = calculate_symbolic_availability(p, y, N, m)

    pq = matrix_multiply_elementwise(p, q)
    pqr = pq * r
    pqry = matrix_multiply_elementwise(pqr, y)
    R = sum(matrix_multiply_elementwise(a, pqry))

    # simplify the expression
    return expand(simplify(R)), (r, q, p, y)

def convert_to_linear(r, y):
    # list of terms that need substitution
    need_substitution = []

    for element in Add.make_args(r):
        # element is r*p_1*q_1*Y_1, for example
        # saw_one_Y is a flag if we've seen on Y yet
        saw_one_Y = False
        for var in element.args:
            # var is just r or p_1 or q_1 or Y_1
            if var in y:
                # if we've already seen, we'll need an X
                if saw_one_Y:
                    need_substitution.append(element)
                    break
                # else, raise the flag
                else:
                    saw_one_Y = True

    # create symbolic variables
    x = Matrix(symbols(f'X1:{len(need_substitution)+1}'))

    # mapping from X to counterpart Y
    X_to_Y = dict()

    for i, need_sub in enumerate(need_substitution):
        ys = 1
        for var in need_sub.args:
            if var in y:
                ys *= var
        X_to_Y[x[i]] = ys

    # return a substituted expression from Y to X and the mapping
    # POTENTIAL BUG: the order of the substitution matters, hence the reversed
    # (we want the most complicated term to be substituted first)
    return r.subs([(val, key) for key, val in reversed(X_to_Y.items())]), X_to_Y

def solve_lp(r_sub, X_to_Y, P_val, Q_val, R_val, variables, binary_y, debug=False):
    '''
    Formulate and solve the LP.
    '''
    m = P_val.size

    r, q, p, y = variables
    x = list(X_to_Y.keys())
    # substitute p, q, r values into the expression
    r_sub = r_sub.subs([(r, R_val[0])])
    r_sub = r_sub.subs([(p[i], P_val[i]) for i in range(m)])
    r_sub = r_sub.subs([(q[i], Q_val[i]) for i in range(m)])
    
    # create the LP model
    nrmp_model = LpProblem("Nurse Revenue Management Problem", LpMaximize)

    num_X = len(X_to_Y)
    
    # create decision variables

    # TODO: make sure X's bounded below by 0
    X = LpVariable.dicts(
        "X", list(range(1, num_X+1)), lowBound=0, upBound=1, cat=LpContinuous # could be binary as well?
    )

    Y = LpVariable.dicts(
        "Y", list(range(1, m+1)), lowBound=0, upBound=1, cat=LpBinary if binary_y else LpContinuous
    )

    def sympy_to_pulp(variable):
        '''
        Converts a sympy X1 to a pulp X1.
        '''
        idx = int(str(variable)[1]) # get the index, that is for X1, get 1
        if variable in x:
            return X[idx]
        elif variable in y:
            return Y[idx]
        else:
            raise Exception('Failed to find variable!')        

    # create objective function
    try:
        nrmp_model += lpSum([element.args[0] * sympy_to_pulp(element.args[1]) for element in Add.make_args(r_sub)])
    except:
        print('ERROR CREATING OBJECTIVE FUNCTION')
        print(list(Add.make_args(r_sub)))

    # add constraints

    # for every X/Y mapping...
    for key, val in X_to_Y.items():
        y_vals = list(val.args)

        # X/Y AND constraint - make sure X can't be 0 when associated Y's all on
        nrmp_model += (
            lpSum([sympy_to_pulp(element) for element in y_vals]) - sympy_to_pulp(key) <= len(y_vals) - 1
        )

        # X/Y OR constraints - make sure that when X is 1, all associated Y's are 1
        for y_val in y_vals:
            nrmp_model += (
                sympy_to_pulp(y_val) - sympy_to_pulp(key) >= 0
            )

    # msg=0 part suppresses output
    solver = GUROBI_CMD(msg=0)
    nrmp_model.solve(solver)

    # get the solution
    soln = np.zeros(m, dtype=np.int64)
    for i in range(m):
        if Y[i + 1].value() == 1:
            soln[i] = 1

    if debug:
        # return objective function value so we can compare to actual
        x_soln = np.zeros(num_X)
        for i in range(num_X):
            if X[i + 1].value() == 1:
                x_soln[i] = 1
        
        r_sub = r_sub.subs([(y[i], soln[i]) for i in range(m)])
        r_sub = r_sub.subs([(x[i], x_soln[i]) for i in range(num_X)])
        print(r_sub)

    return soln.reshape((m, 1))

def MIPCol(pi: ProblemInstance, j: int, binary_y=True):
    # suppress "Spaces are not permitted in the name" warning from PULP
    warnings.simplefilter("ignore")
    R, variables = get_revenue_function(pi.m, pi.N[j])
    R_sub, X_to_Y = convert_to_linear(R, variables[3]) # variables[3] is y
    return solve_lp(R_sub, X_to_Y, pi.P[:, j], pi.Q[:, j], pi.R[j], variables, binary_y)
    

MIP = col_aggregator(MIPCol)