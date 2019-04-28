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

class RiemannianSolver: '''Stochastic Riemannian Eigensolver
        
        Args:
          batchsize: number of data sampled for each udpate
          npass: aka epoch, number of passes through the complete data

          stepsize_type:
          stepsize_init: 
          stepsize_lambda:

    '''
    def __init__(self, A, k, **opt):
        self.batchsize = setOption(100, opt.get('batchsize'), lambda x: x > 0)
        self.npass = setOption(10, opt.get('npass'), lambda x: x > 0)
        self.stepsize_type = setOption('decay', opt.get('stepsize_type'), lambda x: type(x) is str and x in ['decay', 'fixed', 'hybrid', 'KHA/et'])
        self.stepsize_init = setOption(5, opt.get('stepsize_init'), lambda x: x >= 0)
        self.stepsize_lambda = setOption(1e-3, opt.get('stepsize_lambda'), lambda x: x >= 0)
        
        self.inneriter = setOption(20, opt.get('inneriter'), lambda x: x > 0)
        self.checkperiod = setOption(self.inneriter, opt.get('checkperiod'), lambda x: type(x) is int and x > 0)
        
        self.vars = {} # This 'catch-all' dict is reserved for situations when inheriting class needs to get variable that isn't observed.

        self.A = A
        self.n, self.m = A.shape
        self.k = k
        self.inneriter = math.ceil(self.n / self.batchsize)
        self.checkperiod = setOption(self.inneriter, opt.get('checkperiod'), lambda x: type(x) is int and x > 0)
        
    def train(self, X=None, recorder=None):
        self.initialize()
        
        if X is None:
            X = self.init_X()
        
        X_update_operator = self.get_update_operator(X)
        info = {'loss_hist': [], 'time': []}
        
        if recorder is None:
            recorder = {}
            
        for key in recorder:
            if key not in info:
                info[key] = []
            
        print('Iter                 Loss            Time')
    
        for epoch in range(1,self.npass+1):
            indexer = self.getIndexer()

            for t in range(self.inneriter):
                start = time()
                global_t = (epoch - 1)*self.inneriter + t

                batch = indexer(t)
                
                G = self.getGradient(X, batch)
                
                X = X_update_operator(X, -G)


                if global_t > 0 and global_t % self.checkperiod == 0: 
                    time_elapsed = time() - start
                    info['loss_hist'].append(self.getCost(X))
                    info['time'].append(global_t)
                    
                    for stat, func in recorder.items():
                        info[stat].append(recorder[stat](X))

                    print('{0:4d}         {1:10e}        {2:5f}'.format(info['time'][-1], info['loss_hist'][-1], time_elapsed))
                    
                self.vars = {} # clear all hidden variables here
    
        return X, info
        
    
    def get_update_operator(self, X):
        
        if self.stepsize_type == 'KHA/et':
            X_update_operator = optimize.KHAet(X)
        
        else:
            X_update_operator = optimize.Vanilla(stepsize_init=self.stepsize_init, stepsize_type=self.stepsize_type, stepsize_lambda=self.stepsize_lambda)
        return X_update_operator


    def init_X(self): 
        X = np.random.randn(self.m, self.k)
        X, _ = np.linalg.qr(X, 'reduced')
        return X

    
    def get_indexer(self):
        perm = np.random.permutation(self.n)
        
        def get_batch(t):
            lowr = t*self.batchsize
            uppr = np.minimum((t+1)*self.batchsize, self.n)
            return perm[lowr: uppr]
            
        return get_batch    
    
            
    def get_cost(self, X):
        return self.vars['batch_cost']
    
    
    def get_gradient(self, X, batch):
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


def rlbdm(A, k, n_samples, solver_func):
    n = A.shape[0]
    colperm = np.random.choice(n, l, replace=False)
    C = A[:, colperm]
    
    X, info = solver_func(C, k)
    
    rep_labels = KMeans(n_clusters=k, n_init=10, max_iter=10, n_jobs=-1, random_state=0).fit_predict(X)
    
    knn = np.argmax(C, axis=1)
    
    labels = np.zeros(n, dtype=np.int32)
    for i in range(n):
        labels[i] = rep_labels[knn[i]]
    
    return X, labels, info, C
