import numpy as np
from pylab import *

def knn_search(x, data, K):
    D= data.shape[0] # number of data
    K = K if K < D else D # K-nearest neighbors may not be more than data points given

    # euclidean distance
    dist = np.sqrt(((data - x)**2).sum(axis=1))
    idx = np.argsort(dist) # sorting
    # return the indices of K nearest neighbors
    return idx[:K]

def LLE(data, k, d=2):
    K = k
    N, D = data.shape
    tol = 1e-3 if K > D else 0 # regularization if K > D

    W = np.zeros((N,N))
    for i in range(N):
        # find neighborhood for each datapoint
        # idx of k Nearest neighbors
        nbh_idx = knn_search(data[i,:],data, K+1)[1:] # K+1 as one finding will be Xi itself

        # compute weigths
        Z = data[nbh_idx,:] # kNN samples
        Z = Z - data[i,:] # substract Xi from every neighbor
        C = np.dot(Z,Z.T) # local covariance - K x K
        C = C + identity(K) * tol * trace(C) # regularization

        w = np.linalg.solve(C,1) # solve Cw=1
        w = w/float(sum(w))
        W[i,nbh_idx] = w.reshape((K,)) # set Wij = w/sum(w) for every neighbor

    M = identity(N) - W
    M = M.T * M

    # compute embedding
    eig_vals, eig_vecs = np.linalg.eig(M)
    i = np.argsort(eig_vals)[::-1] # sort by magnitude desc
    eig_vals = eig_vals[i]
    eig_vecs = eig_vecs[:, i]
    Y = eig_vecs[:, eig_vals != 0][:, 1:d+1] # select first d non-zero eig_vec
    return Y
