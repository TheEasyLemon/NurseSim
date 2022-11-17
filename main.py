'''
Driver code for finding the expected revenue for the revenue maximization problem.

Dawson Ren, 11/14/22
'''
from ProblemReader.ProblemReader import loadProblemInstance

def main():
    pi = loadProblemInstance('tests/test1_data.txt')
    print(pi.expectedRevenue())

if __name__ == '__main__':
    main()
