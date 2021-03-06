from numpy import median
from scipy.spatial.distance import pdist
import numpy as np
import scipy.sparse as sp
import warnings

def eudist(A, B, squared=True):
    """do not square root if used for gaussian kernel"""
    '''Fast method to create euclidean distance matrix between two sets of
    vectors

    Args:
        A, B:
        squared: whether the euclidean is squared
    
    '''
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
    if squared:
        return (AA - AB + BB)
    else: return (AA - AB + BB) ** 0.5

def gaussianize(fea, sigma=None, gamma=None):
    """\exp(-||x-y||^2 / (2*sigma^2)) or \exp(-gamma*||x-y||^2)"""
#     assert type(fea) == np.ndarray
    if (sigma is not None) and (gamma is not None): raise Error('please use either sigma or gamma but not both')
    if sigma is None and gamma is None: sigma = 1
        
    w = eudist(fea, fea, True)
    if sigma is not None:
        w = np.exp(-w / (2 * sigma ** 2))
    elif gamma is not None:
        w = np.exp(-gamma * w)
    return w    

def laplacianize(W, regularizer=0):
    """Get the *normalized* Laplacian of a symmetric matrix.
    
    Args:
        W: the affinity matrix. It should be symmetric.
        regularizer: the constant added to inflate degree of each node. It can increase statistical performance.
            See: "Understanding Regularized Spectral Clustering via Graph Conductance" https://arxiv.org/abs/1806.01468

    Returns;
        L: The normalized Laplacian matrix $D^{-1/2}WD^{-1/2}$.
        D: The degree of nodes in 1d array.
    """
    D = W.sum(axis=1)
    
    if regularizer > 0:
        D += regularizer
        
    elif regularizer < 0:
        D += D.mean()
        
    if np.any(D <= 0):
        warnings.warn('non-positive degree encountered')
        D = np.maximum(D, 1e-12)
    L = W * (D**(-.5))[:,None] * (D**(-.5))[None,:]
    return L, D
    

def cumdist_matrix(matrix, axis=0):
    """convert matrix to row-cumulative matrix, where each row is a cdf (last entry is 1)
        good for numpy"""
    if axis is 1:
        cum_mat = np.array(list(itertools.accumulate(matrix)))
    elif axis is 0: cum_mat = np.array(list(itertools.accumulate(matrix.transpose()))).transpose()
    else: raise Exception('cumulative matrix only supports first 2 dimensions')
    return cum_mat


def get_sigma(fea):
    """Get the hyper-parameter sigma for RBF kernel of the dataset. 
    It is the median of all distances"""
    
    if fea.shape[0] > 4000:
        rperm = np.random.permutation(fea.shape[0])[:4000]
        feasample = fea[rperm,:]
        
    else:
        feasample = fea
        
    D = pdist(feasample, metric='euclidean')
    
    return median(D)
    
