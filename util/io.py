import random
import numpy as np
from util import matop
import scipy.sparse as sparse
import itertools
import util.rand
from scipy.optimize import linear_sum_assignment

def text_feeder(path, window_size):
    f = open(path, 'rb')
    word_list = [word for line in f for word in line.split()]
    word_set = set(word_list)
    word_to_idx = dict()
    idx = 0
    for word in word_set:
        word_to_idx[word] = idx
        idx += 1
    idx_list = [word_to_idx[word] for word in word_list]
    while True:
        for i in range(window_size, len(word_list) - window_size):
            for j in range(window_size * 2 + 1):
                yield ((idx_list[i], idx_list[i - window_size + j]))


def make_walk(adjmatrix, v_size, node, walk_length):
    walk = [node]
    if not sparse.isspmatrix(adjmatrix):
        for _ in range(walk_length-1):
            node = random.choices(range(v_size), cum_weights=adjmatrix[node], k=1)[0]
            walk.append(node)
    else:
        for _ in range(walk_length-1):
            _,c,v = sparse.find(adjmatrix[node])
            node = random.choices(c, weights=v, k=1)[0]
            walk.append(node)
    return walk


def graph_feeder(adjmatrix_path=None, adjmatrix=None, walk_length=40, window_size=40):
    if not adjmatrix_path is None and adjmatrix is None:
        adjmatrix = np.loadtxt(adjmatrix_path)
    elif (adjmatrix_path is None and adjmatrix is None) or (adjmatrix_path is not None and adjmatrix is not None):
        raise Exception('either adjmatrix_path or adjmatrix but not both has to be specified')
    n = np.size(adjmatrix, 0)
    m = np.size(adjmatrix, 1)
    if not n == m: raise Exception('has to be square matrix')
    # skip_window = 2 * window_size

    sample_generator = util.rand.alias_sampling(adjmatrix)
    while True:
        od = np.arange(n)
        random.shuffle(od)
        for node in od:
            walk = sample_generator.sample_walk(node, walk_length)
            for i in range(walk_length):
                for _ in range(window_size):
                    left_gap = i if i < window_size else window_size
                    for t in walk[i - left_gap:i]:
                        yield(node, t)
                    right_gap = walk_length-i-1 if walk_length-i-1 < window_size else window_size
                    for t in walk[i+1:i+right_gap+1]:
                        yield(node, t)

def batch_feeder(path=None, mat=None, mode='graph', batch_size=128, walk_length=40, window_size=10):
    if mode == 'text':
        feed = text_feeder(path, window_size)
    elif mode == 'graph':
        feed = graph_feeder(path, mat, walk_length, window_size)
    else: raise Exception('unsupported mode')

    context = np.ndarray([batch_size], dtype=np.int32)
    targets = np.ndarray([batch_size, 1], dtype=np.int32)
    while True:
        for i in range(batch_size):
            context[i], targets[i, 0] = next(feed)
        yield (context, targets)

def greedy_matching(labels, targets, in_place=False):
    """match labels to target labels, so that difference is minimized"""
    from collections import Counter
    labels_count = Counter(labels)
    targets_count = Counter(targets)
    labels_sort = labels_count.most_common()
    print(labels_sort)
    targets_sort = targets_count.most_common()
    print(targets_sort)
    mapping = dict()
    diff = 0
    for x, y in zip(labels_sort, targets_sort):
        mapping[x[0]] = y[0]
        diff += abs(x[1] - y[1])
    print(mapping)
    return np.array([mapping[x] for x in labels]), diff

def accuracy(label, ground):
    n = label.size
    return (label.flat == ground.flat).sum() / n

def best_map(labels, targets):
    """wrapper for scipy.optimize.linear_sum_assignment (which uses Hungarian algorithm, O(n^3) or O(n^4)?)
        NOTE: It is implicit that targets and labels unique values are in range(1, k), where k is number
            of unique labels"""

    n = len(set(targets))
    m = len(set(labels))
    if not n == m: raise Warning('number of unique labels do not match. This will result in unassigned labels')
    w = np.zeros((n,m), dtype=np.int64)
    for i,j in zip(targets, labels):
        w[i-1,j-1] -= 1
    row_ind, col_ind = linear_sum_assignment(w)
    corrected_labels = col_ind[labels - 1] + 1
    return corrected_labels, w[row_ind, col_ind].sum()

def make_weight_matrix(fea, mode, **kwargs):
    assert type(fea) == np.ndarray
    if mode == 'gaussian':
        #make weight matrix (faster version)
        w = matop.eudist(fea, fea, False)
        #use gaussian affinity
        w = np.exp(-w / (2 * kwargs['sigma'] ** 2))
    elif mode == 'cosine':
        fea = fea / (np.sum(fea ** 2, 1) ** 0.5).reshape([np.size(fea, 0), 1])
        w = fea * fea.transpose()
    else:
        raise Exception('unsupported mode: {}'.format(mode))
    return w
