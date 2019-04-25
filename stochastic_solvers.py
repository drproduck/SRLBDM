from myutils import *
import math
import numpy as np
import optimize
from mylinalg import *
from time import time

class StochasticOptimizer:
    
    def __init__(self, **opt):
        
        self.batchsize = setOption(100, opt.get('batchsize'), lambda x: x > 0)
        self.npass = setOption(10, opt.get('npass'), lambda x: x > 0)
        self.stepsize_type = setOption('decay', opt.get('stepsize_type'), lambda x: type(x) is str and x in ['decay', 'fixed', 'hybrid', 'KHA/et'])
        self.stepsize_init = setOption(5, opt.get('stepsize_init'), lambda x: x >= 0)
        self.stepsize_lambda = setOption(1e-3, opt.get('stepsize_lambda'), lambda x: x >= 0)
        
        self.inneriter = setOption(20, opt.get('inneriter'), lambda x: x > 0)
        self.checkperiod = setOption(self.inneriter, opt.get('checkperiod'), lambda x: type(x) is int and x > 0)
        
        self.vars = {} # This 'catch-all' dict is reserved for situations when inheriting class needs to get variable that isn't observed.
        
    def init_X(self):
        pass
    
    def initialize(self):
        """Prepare all other variables needed when training. 
        For example, in adaptive step size, we often need to store past gradients or their statistics
        
        """
        pass
    
    def getUpdateOperator(self, X):
        if self.stepsize_type == 'KHA/et':
            X_update_operator = optimize.KHAet(X)
        
        else:
            X_update_operator = optimize.Vanilla(stepsize_init=self.stepsize_init, stepsize_type=self.stepsize_type, stepsize_lambda=self.stepsize_lambda)
        return X_update_operator
    
    def getGradient(self, X, batch):
        pass
    
    def postProcessG(self, G, X):
        return G
    
    def getCost(self, X):
        pass
    
    def getIndexer(self):
        pass
    
    def train(self, X=None, recorder=None):
        self.initialize()
        
        if X is None:
            X = self.init_X()
        
        X_update_operator = self.getUpdateOperator(X)
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
                
                G = self.postProcessG(G, X)
                
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
    
    
    
