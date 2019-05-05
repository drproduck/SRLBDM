#!/usr/bin/env python

import traceback
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons,make_circles, make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from time import time
import kernel
import optimize
import scipy.sparse as sp
import numpy as np
from scipy.linalg import eigh
import math
from util.helper import setOption, load_matlab

from util.metrics import get_y_preds

from RiemannSolver import solver
        

def get_affinity_matrix(data, landmarks, sigma):
    '''Create RBF kernel matrix between data and landmarks
    '''
    dist_mat = kernel.eudist(data, landmarks, squared=True)
    return np.exp(-dist_mat / (2 * sigma ** 2))



def create_batch_feeder(data, landmarks, sigma, batchsize, repeat):
    '''A batch feeder is an iterator that yields new batch every call and opens
    new file when needed. Also make sure that batch is always batchsize-large
    '''
    n_samples = data.shape[0]
    perm = []
    for _ in range(repeat):
        perm += np.random.permutation(n_samples).tolist()

    i = 0
    while i < n_samples * repeat:
        data_batch = get_affinity_matrix(data[perm[i : i+batchsize], :], landmarks, sigma)
        yield(data_batch, np.argmax(data_batch, axis=1).flatten().tolist())
        i = i + batchsize




data, true_labels = load_matlab('data/mnist.mat')
n = data.shape[0]

n_landmarks = 500
n_iters = 200
batchsize = 500
repeat = math.ceil(n_iters * batchsize / n)
n_clusters = 10

landmarks = data[np.random.choice(n, n_landmarks, replace=False), :]

print('Preprocessing..')
sigma = kernel.get_sigma(data)

batch_feeder = create_batch_feeder(data, landmarks, sigma, batchsize, repeat)

rieman_opt = solver(n_landmarks, n_clusters, n_iters, stepsize_init=0.001, checkperiod=5)

print('Training..')
X, knn, info = rieman_opt.train(batch_feeder)

print('Postprocessing..')
#rep_labels = KMeans(n_clusters=n_clusters, n_init=10, max_iter=10, n_jobs=-1, random_state=0).fit_predict(X)
# try mini batch
rep_labels = MiniBatchKMeans(n_clusters=n_clusters, max_iter=10, batch_size=100, n_init=10).fit_predict(X)

labels = np.zeros(n, dtype=np.int32)

for i in range(n):
    labels[i] = rep_labels[knn[i]]

labels, _ = get_y_preds(labels, true_labels, n_clusters)

# accuracy =(labels == true_labels).astype(int).sum()
accuracy = 0
for i in range(len(labels)):
    accuracy += (labels[i] == true_labels[i])

accuracy = accuracy / len(labels)
print(accuracy)
