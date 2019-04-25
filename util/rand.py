from random import random
import numpy as np
from math import floor
import scipy.sparse as sparse

class alias_sampling():
    def __init__(self, array):
        s = np.shape(array)
        self.shape = s
        self.issparse = sparse.isspmatrix(array)
        if len(s) != 1 and len(s) != 2: raise Exception('Invalid shape, only array of dimension 1 or 2 is allowed')
        self.n_face = s[0] if len(s) == 1 else s[1]
        self.weights_table = array
        if self.issparse:
            r,c,_ = sparse.find(self.weights_table)
            self.prob_table = sparse.coo_matrix(([0.0]*c.size, (r,c)), shape=self.shape,
                                                dtype=np.float64).tocsc()
            self.alias_table = sparse.coo_matrix(([0]*c.size, (r,c)), shape=self.shape,
                                                 dtype=np.int32).tocsc()
        else:
            self.prob_table = np.zeros(shape=s, dtype=np.float64)
            self.alias_table = np.zeros(shape=s, dtype=np.int64)
        if len(s) == 1: self.dim1_sampling(self.weights_table, self.prob_table,
                                           self.alias_table)
        elif len(s) == 2: self.dim2_sampling()


    def dim1_sampling(self, weights, prob, alias):
        alpha = (1.0 * self.n_face) / weights.sum()
        weights *= alpha
        indices = range(weights.shape[0])

        small = list()
        large = list()
        for i in indices:
            small.append(i) if weights[i] < 1 else large.append(i)
        while len(small) != 0 and len(large) != 0:
            s = small.pop()
            l = large.pop()
            prob[s] = weights[s]
            alias[s] = l
            weights[l] = weights[s] + weights[l] - 1
            small.append(l) if weights[l] < 1 else large.append(l)

        while len(large) != 0:
            g = large.pop()
            prob[g] = 1
        while len(small) != 0:
            g = small.pop()
            prob[g] = 1

    def dim1_sparse_sampling(self, index):
        _,c,_ = sparse.find(self.weights_table[index])
        nz_face = np.size(c)
        alpha = (1.0 * nz_face) / self.weights_table[index].sum()
        self.weights_table[index] *= alpha
        small = list()
        large = list()
        indices = c
        for i in indices:
            small.append(i) if self.weights_table[0,i] < 1 else large.append(i)
        while len(small) != 0 and len(large) != 0:
            s = small.pop()
            l = large.pop()
            self.prob_table[0,s] = self.weights_table[0,s]
            self.alias_table[0,s] = l
            self.weights_table[0,l] = self.weights_table[0,s] + self.weights_table[0,l] - 1
            small.append(l) if self.weights_table[0,l] < 1 else large.append(l)

        while len(large) != 0:
            g = large.pop()
            self.prob_table[0,g] = 1
        while len(small) != 0:
            g = small.pop()
            self.prob_table[0,g] = 1

    def dim2_sampling(self):
        if self.issparse:
            for i in range(self.weights_table.shape[0]):
                self.dim1_sparse_sampling(i)
        else:
            for i in range(self.weights_table.shape[0]):
                self.dim1_sampling(self.weights_table[i], self.prob_table[i],
                                   self.alias_table[i])

    def sample(self, index=None):
        if len(self.shape) == 2 and index is None: raise Exception('index required for table of multiple distributions')
        if self.issparse:
            _,c,_ = sparse.find(self.weights_table[index])
            face = c[floor(random()*c.size)]
            return face if random() < self.prob_table[index,face] else self.alias_table[index,face]
        else:
            if len(self.shape) == 1:
                prob = self.prob_table
                alias = self.alias_table
            elif len(self.shape) == 2:
                prob = self.prob_table[index,:]
                alias = self.alias_table[index,:]
            face = floor(random()*self.n_face)
            return face if random() < prob[face] else alias[face]

    def sample_walk(self, start_node, walk_length=40):
        if start_node is None and len(self.shape) != 2:
            raise Exception('walk only generated from adjacency matrix. Are you sure input is ndarray with 2 dimension > 1?')
        ret = list()
        for _ in range(walk_length):
            next_node = self.sample(start_node)
            ret.append(next_node)
            node = next_node
        return ret

def main():
    from time import time
    ar = np.random.rand(5000, 5000)
    ar[np.where(ar > 0.2)] = 0
    a = time()
    gen = alias_sampling(ar)
    x = np.array([gen.sample(0) for i in range(1000)])
    b = time()
    print('time for dense: {}'.format(b-a))
    a = time()
    gen_sparse = alias_sampling(sparse.csc_matrix(ar))
    z = np.array([gen_sparse.sample(0) for _ in range(1000)])
    b = time()
    print('time for sparse: {}'.format(b-a))
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.subplot(121)
    plt.hist(x)
    plt.subplot(122)
    plt.hist(z)
    plt.show()

if __name__ == '__main__':
    main()