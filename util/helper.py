from scipy.spatial.distance import pdist
from numpy.random import permutation
from numpy import median
import numpy as np

def getSigma(fea):
    """Get the hyper-parameter sigma for RBF kernel of the dataset. 
    It is the median of all distances"""
    
    if fea.shape[0] > 4000:
        rperm = permutation(fea.shape[0])[:4000]
        feasample = fea[rperm,:]
        
    else:
        feasample = fea
        
    D = pdist(feasample, metric='euclidean')
    
    return median(D)
    
    
def setOption(default, user_input, isValid):
    """Merge 2 options, prioritizing the second (credit to manopt)"""
    
    if user_input is None or (isValid is not None and not isValid(user_input)):
        return default
    else: return user_input
    
    
def mergeOptions(opt1, opt2):
    """Merge 2 dictionaries, prioritizing the second
    """
    newopt = copy.copy(opt2)
    print(type(newopt))
    for k, v in opt1.items():
        if k not in opt2:
            newopt[k] = v
            
    return opt2

    
# def stableNormalize(X, check_zero=True):
#     """stable L2 normalization of a vector, or the last dimension of a numpy array. In-place
    
#     Args:
#         X: a numpy array
        
#     """
    
#     if len(shape(X)) == 1:
#         if check_zero:
#             if np.all(X == 0):
#                 return X
#         else:
#             return X / np.linalg.norm(X)
        
#     elif len(shape(X) == 2:
#         if check_zero:
#             idx = np.where(np.all(X == 0, axis=1) == 0) # indices that can be normalized
#             norm = np.sqrt(np.sum(X[idx]**2, 1))
#             return X 
        
#         norm = np.maximum(norm, 1e-15)
#         X = X / norm[:, None]