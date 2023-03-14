import numpy as np
# serious performance gains from eliminating combinations...
from itertools import combinations

cimport numpy as np
cimport cython

np.import_array()

# get malloc, free, calloc from C stdlib
cdef extern from "stdlib.h":
    ctypedef int intptr_t
    void* malloc(size_t size)
    void free(void* ptr)

cdef void generate_combinations(int n, int k, int start, int depth, int* combination, double[:] Py, double* prob):
    cdef int i, j, combo_idx, zero_flag
    cdef double prod
    if depth == k:
        prod = 1
        # index of combination
        combo_idx = 0
        zero_flag = 0
        # these are the 1s
        for i in range(n):
            if combo_idx < k and i == combination[combo_idx]:
                combo_idx += 1
                if Py[i] == 0:
                    zero_flag = 1
                    break
                prod *= Py[i]
            else:
                if Py[i] == 1:
                    zero_flag = 1
                    break
                prod *= (1 - Py[i])
        if zero_flag == 0:
            prob[0] += prod
        return
    for i in range(start, n):
        combination[depth] = i
        generate_combinations(n, k, i + 1, depth + 1, combination, Py, prob) 

cpdef double get_availability(int n, int k, double[:] Py):
    # Associated code in 
    # for tup in combinations(range(i), n):
    #     prob = 1
    #     for k in range(i):
    #         if k in tup:
    #             prob *= Py[k]
    #         else:
    #             prob *= (1 - Py[k])
    #     A[i, n] += prob

    cdef int* combination = <int*>malloc(k * sizeof(int))
    cdef double* prob = <double*>malloc(sizeof(double))
    prob[0] = 0 # set to 0 to begin

    generate_combinations(n, k, 0, 0, combination, Py, prob)

    free(combination)

    return prob[0]

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
cpdef calculateAvailabilityCol(long N, long m, long[:] y, double[:] p):
    '''
    Takes the following arguments:
    - y - the policy we are considering
    - m - the number of nurses
    - N - the number of nurses we can schedule
    - p - the probability each nurse will show up to the shift
    Returns:
    - a np.ndarray of the same shape as y and p
      with the probability the shift will be available to that nurse
    '''
    cdef long n, i, j # type the loop indices

    cdef double[:, ::1] A = np.triu(np.ones((m, N)), 0) # the row gives the nurse, the column gives the availability when j shifts allowed
    cdef double[:] Py = np.zeros(m)
    for i in range(m):
        Py[i] = p[i] * y[i]

    # first fill in 0th column of A
    for i in range(1, m):
        A[i, 0] = (1 - Py[i - 1]) * A[i - 1, 0]

    for n in range(1, N):
        for i in range(n + 1, m): # for every nurse, starting after the entries guaranted to be 1
            # Use optimized C code
            A[i, n] = get_availability(<int> i, <int> n, Py)

    cdef double[:] A_col = np.zeros(m)
    cdef double acc_sum

    # sum across rows, make sure first N get 1
    # because first N nurses always have a shift available
    for i in range(m):
        if i < N:
            A_col[i] = 1
        else:
            acc_sum = 0
            for j in range(N):
                acc_sum += A[i, j]
            A_col[i] = acc_sum
    
    return A_col