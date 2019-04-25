import numpy as np
import time
import itertools
import scipy.sparse as sp
def sloweudist(A, B):
    n = np.size(A, 0)
    m = np.size(B,0)
    x = np.zeros(shape=[n,m])
    for i in range(n):
        for j in range(m):
            x[i,j] = np.linalg.norm(A[i,:] - B[j,:])
    return x

def eudist(A, B, sqrted=True):
    n, n2 = A.shape
    m, m2 = B.shape
    assert(n2 == m2)
    if sp.isspmatrix(A):
        a = np.asarray(np.sum(A.multiply(A), 1)).reshape([n,1])
    else: a = np.sum(A ** 2, 1).reshape([n,1])
    AA = np.repeat(a, m, 1)
    if sp.isspmatrix(B):
        b = np.asarray(np.sum(B.multiply(B), 1)).reshape([1,m])
    else: b = np.sum(B ** 2, 1).reshape([1,m])
    BB = np.repeat(b, n, 0)
    AB = 2 * A.dot(B.T)
    if sqrted:
        return (AA - AB + BB) ** 0.5
    else: return AA - AB + BB

def cumdist_matrix(matrix, axis=0):
    """convert matrix to row-cumulative matrix, where each row is a cdf (last entry is 1)
        good for numpy"""
    if axis is 1:
        cum_mat = np.array(list(itertools.accumulate(matrix)))
    elif axis is 0: cum_mat = np.array(list(itertools.accumulate(matrix.transpose()))).transpose()
    else: raise Exception('cumulative matrix only supports first 2 dimensions')
    return cum_mat

def main():
    a = np.array([[1,2],[3,4]])
    # b = np.array([[5,6],[7,8]])
    # print(eudist(a,b,False))
    #
    # a = np.random.normal(0,1, [2000, 2000])
    # b = np.random.normal(0,1, [4000, 2000])
    # start = time.time()
    # x = eudist(a,b, True)
    # stop = time.time()
    # print(x)
    # print(stop - start)
    # start = time.time()
    # x = sloweudist(a,b)
    # stop = time.time()
    # print(x)
    # print(stop - start)
    x = cumdist_matrix(a, 0)
    print(x)
    print(cumdist_matrix(a, 1))

if __name__ == "__main__":
    main()
