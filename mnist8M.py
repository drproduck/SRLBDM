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

from RiemannSolver import solver
        

def get_affinity_matrix(data, landmarks, sigma):
    '''Create RBF kernel matrix between data and landmarks
    '''
    dist_mat = kernel.eudist(data, landmarks, squared=True)
    return np.exp(-dist_mat / (2 * sigma ** 2))

def gather_mnist_labels(data_pat):
    '''Gather all labels from data files
    '''
    size = int(1e5)
    n_chunks = 1
    labels = np.zeros(size*n_chunks, dtype=np.int)
    for i in range(n_chunks):
        with np.load(data_pat.format(i)) as data:
            labels[i*size : (i+1)*size] = data['label']
    return labels


def create_batch_feeder(data_pat, landmarks, sigma, batchsize):
    '''A batch feeder is an iterator that yields new batch every call and opens
    new file when needed. Also make sure that batch is always batchsize-large
    '''

    leftover_batch = None
    n_chunks = 1
    for i in range(n_chunks):
        print('Loading new data file..')
        with np.load(data_pat.format(i)) as data:
            A = data['data']
            # A = get_affinity_matrix(A, landmarks, sigma)
            n_samples = A.shape[0]
            perm = np.random.permutation(n_samples).tolist()

            offset = 0
            if leftover_batch is not None:
                offset = batchsize - leftover_batch.shape[0]
                A_batch_index = perm[0 : offset]
                A_batch = get_affinity_matrix(A[A_batch_index, :], landmarks, sigma)
                A_batch = np.vstack((leftover_batch, A_batch).flatten().tolist())
                if A_batch.shape[0] == batchsize:
                    yield (A_batch, np.argmax(A_batch, axis=1))
                else: 
                    leftover_batch = A_batch

            j = offset
            while n_samples > j:
                A_batch_index = perm[j : j+batchsize]
                A_batch = get_affinity_matrix(A[A_batch_index, :], landmarks, sigma)
                if A_batch.shape[0] == batchsize:
                    leftover_batch = None
                    j = j + batchsize
                    yield (A_batch, np.argmax(A_batch, axis=1).flatten().tolist())
                else:
                    leftover_batch = A_batch
                    break




n_chunks = 1
chunksize = int(1e5)
n = n_chunks * chunksize
batchsize = 10000
n_clusters = 10
n_iters = int(n / batchsize)
data_pat = '../mnist8m_dataset/data_batch_{}.npz'

landmarks = np.load(data_pat.format(81))['data']
n_landmarks = landmarks.shape[0]

print('Preprocessing..')
sigma = kernel.get_sigma(landmarks)

true_labels = gather_mnist_labels(data_pat)

batch_feeder = create_batch_feeder(data_pat, landmarks, sigma, batchsize)

rieman_opt = solver(n_landmarks, n_clusters, n_iters, stepsize_init=0.0001, checkperiod=5)

print('Training..')
X, knn, info = rieman_opt.train(batch_feeder)

print('Postprocessing..')
rep_labels = KMeans(n_clusters=n_clusters, n_init=10, max_iter=10, n_jobs=-1, random_state=0).fit_predict(X)

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
