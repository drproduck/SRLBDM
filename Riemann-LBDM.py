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

class RiemannianSolver: 
    '''Stochastic Riemannian Eigensolver
        
        Args:
            k: number of clusters
            batchsize: number of data sampled for each udpate
            n_iters: number of iterations

            stepsize_type:
            stepsize_init: 
            stepsize_lambda:

    '''
    def __init__(self, n_clusters, n_iters, **opt):
        self.n_clusters = n_clusters
        self.n_iters = n_iters
        self.batchsize = setOption(100, opt.get('batchsize'), lambda x: x > 0)
        self.stepsize_type = setOption('decay', opt.get('stepsize_type'), lambda x: type(x) is str and x in ['decay', 'fixed', 'hybrid', 'KHA/et'])
        self.stepsize_init = setOption(5, opt.get('stepsize_init'), lambda x: x >= 0)
        self.stepsize_lambda = setOption(1e-3, opt.get('stepsize_lambda'), lambda x: x >= 0)
        
        self.checkperiod = setOption(100, opt.get('checkperiod'), lambda x: type(x) is int and x > 0)

        self.vars = {} # This 'catch-all' dict is reserved for situations when inheriting class needs to get variable that isn't observed.
        

    def train(self, batch_feeder, X=None, recorder=None):
        
        if X is None:
            X = self.init_X()
        
        update_operator = self.get_update_operator(X)
        info = {'loss_hist': [], 'time': []}
        

        print('Iter                 Loss            Time')
    

        for t in range(1, n_iters+1):

            try:
                A_batch = next(batch_feeder)
            except:
                break

            start = time()
                
            G = self.get_gradient(X, A_batch)
            
            X = update_operator(X, -G)

            if global_t > 0 and global_t % self.checkperiod == 0: 
                time_elapsed = time() - start
                info['loss_hist'].append(self.getCost(X))
                info['time'].append(t)
                info['d2I'] = np.linalg.norm(X.T @ (self.vars['B'] * X) - np.eye(self.k), 'fro')
                
                print('{0:4d}         {1:10e}        {2:5f}'.format(info['time'][-1], info['loss_hist'][-1], time_elapsed))
                
            self.vars = {} # clear all hidden variables here

        return X, info

    
    def get_update_operator(self):
          update_operator = optimize.Vanilla(stepsize_init=self.stepsize_init, stepsize_type=self.stepsize_type, stepsize_lambda=self.stepsize_lambda)
        return update_operator


    def init_X(self): 
        X = np.random.randn(self.m, self.k)
        X, _ = np.linalg.qr(X, 'reduced')
        return X
    
            
    def get_cost(self, X):
        return self.vars['batch_cost']
    
    
    def get_gradient(self, X, A_batch):

        batchsize, dim = A_batch.shape[0]

        Dcol = A_batch.sum(axis=0).reshape(dim, 1)
        Drow = A_batch.sum(axis=1).reshape(batchsize, 1)
        normalized_A = A / Drow**0.5
        
        Dcol = Dcol / batchsize
        self.vars['B'] = Dcol

        G = normalized_A @ X # b * m * k
        self.vars['batch_cost'] = np.linalg.norm(G, 'fro')**2 / batchsize
        G = normalized_A.T @ G - D2 * (X @ (G.T @ G)) # which multiplication order is fastest. (m * m) (m * k) (k * b) (b * k)

        return G
        


def get_affinity_matrix(data, landmarks, sigma):
    dist_mat = kernel.eudist(data, landmarks, squared=True)
    return np.exp(-dist_mat / (2 * sigma ** 2))

def gather_mnist_labels(data_pat):
    size = 100000
    labels = np.zeros(size*81, dtype=np.int)
    for i in range(81):
        with load(data_pat.format(i)) as data:
            labels[i*size : (i+1)*size] = data['label']
    return labels


def create_batch_feeder(data_pat, landmarks, sigma, batchsize):

    for i in range(81):
        with load(data_pat.format(i)) as data:
            A = data['data']
            A = get_affinity_matrix(A, landmarks, sigma)
            n_samples = A.shape[0]
            perm = np.random.permutation(n_samples)

            offset = 0
            if leftover_batch is not None:
                offset = batchsize - leftover_batch.shape[0]
                A_batch = np.vstack((leftover_batch, A[0 : offset, :]))
                if A_batch.shape[0] == batchsize:
                    yield A_batch
                else: 
                    leftover_batch = A_batch

            j = offset
            while n_samples > j:
                A_batch = A[j : j+batchsize, :]
                if A_batch.shape[0] == batchsize:
                    leftover_batch = None
                    j = j + batchsize
                    yield A_batch
                else:
                    leftover_batch = A_batch
                    break

n_chunks = 81
chunksize = int(1e5)
batchsize = 10000
n_clusters = 10
n_iters = int(n_chunks * chunksize / batchsize)
data_pat = 'data_batch_{}.npz'
landmarks = np.load(data_pat.format(81))
sigma = myutils.get_sigma(landmarks)

labels = gather_mnist_labels(data_pat)

batch_feeder = create_batch_feeder(data_pat, landmarks, sigma, batchsize)

rieman_opt = RiemannianSolver(n_clusters, n_iters, stepsize_init=0.1)

X, info = rieman_opt.train(batch_feeder)


rep_labels = KMeans(n_clusters=k, n_init=10, max_iter=10, n_jobs=-1, random_state=0).fit_predict(X)

knn = np.argmax(C, axis=1)

labels = np.zeros(n, dtype=np.int32)
for i in range(n):
    labels[i] = rep_labels[knn[i]]

return X, labels, info, C
