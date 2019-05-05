#!/usr/bin/env python

import traceback
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons,make_circles, make_blobs
from sklearn.cluster import KMeans
from time import time
import kernel
import optimize
import scipy.sparse as sp
import numpy as np
from scipy.linalg import eigh
import math
from util.helper import setOption
from util.metrics import get_y_preds


class solver: 
    '''Stochastic Riemannian Eigensolver
        
        Args:
            k: number of clusters
            batchsize: number of data sampled for each udpate
            n_iters: number of iterations

            stepsize_type:
            stepsize_init: 
            stepsize_lambda:
            checkperiod:

    '''
    def __init__(self, n_landmarks, n_clusters, n_iters, **opt):
        self.n_landmarks = n_landmarks
        self.n_clusters = n_clusters
        self.n_iters = n_iters

        self.batchsize = setOption(100, opt.get('batchsize'), lambda x: x > 0)
        self.stepsize_type = setOption('decay', opt.get('stepsize_type'), lambda x: type(x) is str and x in ['decay', 'fixed', 'hybrid', 'KHA/et'])
        self.stepsize_init = setOption(5, opt.get('stepsize_init'), lambda x: x >= 0)
        self.stepsize_lambda = setOption(1e-3, opt.get('stepsize_lambda'), lambda x: x >= 0)
        
        self.checkperiod = setOption(100, opt.get('checkperiod'), lambda x: type(x) is int and x > 0)

        self.vars = {} # This 'catch-all' dict is reserved for situations when inheriting class needs to get variable that isn't observed.
        

    def train(self, batch_feeder, X=None, recorder=None):
        '''Iterate through data batches and update solution

            Args:
                batch_feeder: Iterator that yields next batch
                X: if provided, initial solution
                recorder: dict of functions that may be passed to record
                statistics

            Returns:
                X: final solution
                info: statistics

        '''
        knn = []
        
        if X is None:
            X = self.init_X()
        
        update_operator = self.get_update_operator()
        info = {'loss_hist': [], 'time': []}
        

        print('Iter                 Loss            Time')
    

        for t in range(1, n_iters+1):

            try:
                A_batch, knn_batch = next(batch_feeder)
                knn += knn_batch
            except:
                traceback.print_exc()
                break

            start = time()
                
            t1 = time()
            G = self.get_gradient(X, A_batch)
            
            X = update_operator(X, -G)

            if t % self.checkperiod == 0: 
                time_elapsed = time() - start
                info['loss_hist'].append(self.get_cost(X))
                info['time'].append(t)
                info['d2I'] = np.linalg.norm(X.T @ (self.vars['B'] * X) - np.eye(self.n_clusters), 'fro')
                
                print('{0:4d}         {1:10e}        {2:5f}'.format(info['time'][-1], info['loss_hist'][-1], time_elapsed))
                
            self.vars = {} # clear all hidden variables here

        return X, knn, info

    
    def get_update_operator(self):
        '''Specify how the solution is updated

        Returns:
            update_operator: function that takes previous solution and returns
            new solution
        '''
        update_operator = optimize.Vanilla(stepsize_init=self.stepsize_init, stepsize_type=self.stepsize_type, stepsize_lambda=self.stepsize_lambda)
        return update_operator


    def init_X(self): 
        '''Initialize a valid solution
        '''
        X = np.random.randn(self.n_landmarks, self.n_clusters)
        X, _ = np.linalg.qr(X, 'reduced')
        return X
    
            
    def get_cost(self, X):
        '''Return cost to display
        '''
        return self.vars['batch_cost']
    
    
    def get_gradient(self, X, A_batch):
        '''Compute gradient from solution and batch matrix
        '''

        batchsize, dim = A_batch.shape

        Dcol = A_batch.sum(axis=0).reshape(dim, 1)
        Drow = A_batch.sum(axis=1).reshape(batchsize, 1)
        normalized_A = A_batch / Drow**0.5
        
        Dcol = Dcol / batchsize
        self.vars['B'] = Dcol

        G = normalized_A @ X # b * m * k
        self.vars['batch_cost'] = np.linalg.norm(G, 'fro')**2 / batchsize
        G = normalized_A.T @ G - Dcol * (X @ (G.T @ G)) # which multiplication order is fastest. (m * m) (m * k) (k * b) (b * k)

        return G
        
