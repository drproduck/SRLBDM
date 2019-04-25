import matplotlib.pyplot as plt
from sklearn.datasets import make_moons,make_circles, make_blobs
from sklearn.cluster import KMeans
import time
import kernel
import optimize
import scipy.sparse as sp
import numpy as np
from scipy.linalg import eigh
import math
from myutils import *
from stochastic_solvers import *
from util.metrics import get_accuracyk

class Solver(StochasticOptimizer):
    '''No retraction. Initialized with a valid solution.
    '''
    def __init__(self, A, k, **opt):
        StochasticOptimizer.__init__(self, **opt)
        self.A = A
        self.n, self.m = A.shape
        self.k = k
        self.inneriter = math.ceil(self.n / self.batchsize)
        self.checkperiod = setOption(self.inneriter, opt.get('checkperiod'), lambda x: type(x) is int and x > 0)
        
        
    def init_X(self): 
        X = np.random.randn(self.m, self.k)
        X, _ = np.linalg.qr(X, 'reduced')
        return X

    
    def getIndexer(self):
        perm = np.random.permutation(self.n)
        
        def getBatch(t):
            lowr = t*self.batchsize
            uppr = np.minimum((t+1)*self.batchsize, self.n)
            return perm[lowr: uppr]
            
        return getBatch    
    
            
    def getCost(self, X):
        return self.vars['batch_cost']
    
    
    def getGradient(self, X, batch):
        Abatch = self.A[batch,:]
        batchsize = Abatch.shape[0]

        D2 = Abatch.sum(axis=0).reshape(self.m, 1)
        D1 = Abatch.sum(axis=1).reshape(batchsize, 1)
        normalized_Abatch = Abatch / D1**0.5
        
        D2 = D2 / batchsize
        self.vars['B'] = D2

        G = normalized_Abatch @ X # b * m * k
        self.vars['batch_cost'] = np.linalg.norm(G, 'fro')**2 / batchsize
        G = normalized_Abatch.T @ G - D2 * (X @ (G.T @ G)) # which multiplication order is fastest. (m * m) (m * k) (k * b) (b * k)

        return G
    
    def train(self):
        def distance2I(X):
            return np.linalg.norm(X.T @ (self.vars['B'] * X) - np.eye(self.k), 'fro')
        
        recorder = {'distance to manifold': distance2I}
        return super().train(recorder=recorder)


def RiemannLBDM(A, k, n_samples, solver_func):
    n = A.shape[0]
    colperm = np.random.choice(n, l, replace=False)
    C = A[:, colperm]
    print('column sum: ', C.sum(axis=0))
    
    X, info = solver_func(C, k)
    
    rep_labels = KMeans(n_clusters=k, n_init=10, max_iter=10, n_jobs=-1, random_state=0).fit_predict(X)
    
    knn = np.argmax(C, axis=1)
    
    labels = np.zeros(n, dtype=np.int32)
    for i in range(n):
        labels[i] = rep_labels[knn[i]]
    
    return X, labels, info, C
