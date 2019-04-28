import numpy as np
from util.helper import setOption


class Vanilla:
    """all credits to manopt stepsize_sg"""
    def __init__(self, **opt):
        
        self.type = setOption('decay', opt.get('stepsize_type'), lambda x: type(x) is str)
        self.lr = setOption(1, opt.get('stepsize_init'), lambda x: x > 0)
        self.lmbd = setOption(1e-3, opt.get('stepsize_lambda'), lambda x: x > 0)
        self.decaysteps = setOption(100, opt.get('decaysteps'), lambda x: x > 0)
        self.iter = 1
        
    def __call__(self, X, G):
        if self.type == 'decay':
            stepsize = self.lr / (1 + self.lr*self.lmbd*self.iter)
        elif self.type == 'hybrid':
            # decays for only a few initial iterations
            if self.iter < self.decaysteps:
                stepsize = self.lr / (1 + self.lr*self.lmbd*self.iter)
            else:
                stepsize = self.lr / (1 + self.lr*self.lmbd*self.decaysteps)
        else:
            stepsize = self.lr
            
        self.iter += 1
        
        return X - stepsize * G
        
        
        
class KHAet:        
    """
    Adaptive step size KHA\et
    Fast Iterative Principal Component Analysis
    http://www.jmlr.org/papers/volume8/guenter07a/guenter07a.pdf
    The online-BFGS will be implemented
    """
    
    def __init__(self, X):
        self.X = X
        self.iter = 1
        
    def __call__(self, G, eigvec_norm, eigval, epoch, t, init):
        stepsize = eigvec_norm / eigval * (epoch / (t + epoch) * init)
        self.X -= stepsize * G
        self.iter += 1
    

class Adam:
    def __init__(self,w,descent=True,b1=0.9,b2=0.999,eps=1e-8):
        self.descent = descent
        self.w = w
        self.eta = 0.01
        self.m = 0.0
        self.v = 0.0
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 0
        
    def __call__(self,g):
        self.t += 1
        self.m =  self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g**2
        mh = self.m / (1 - self.b1**self.t)
        vh = self.v / (1 - self.b2**self.t)
        if self.descent:
            self.w -= self.eta / (np.sqrt(vh) + self.eps) * mh
        else: 
            self.w += self.eta / (np.sqrt(vh) + self.eps) * mh
