import numpy as np

def sym(Y):
    return (Y + Y.T) / 2

def guf(Y, B):
    """Generalized polar decomposition of an n-by-p matrix Y.
     X'*B*X is identity.
    """
    
    u, _, v = np.linalg.svd(Y, full_matrices=False)

    ssquare, q = np.linalg.eig(u.T @ (B @ u))
    qsinv = q / np.sqrt(ssquare)
    X = u @ ((qsinv @ q.T) @ v.T); # X'*B*X is identity.
           
    return X
