import numpy as np
import math

def mini_batches(X, Y, batch_size=64, random_seed=42):
    np.random.seed(random_seed)
    m = X.shape[1]
    mini_batches = []
    
    permutation = list(np.random.permutation(m))
    shuffle_X = X[:, permutation]
    shuffle_Y = Y[:, permutation].reshape((1, m))
    
    inc = batch_size
    
    
    num_complete_minibatches = math.floor(m / batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffle_X[:, k*batch_size: (k+1)*batch_size]
        mini_batch_Y = shuffle_Y[:, k*batch_size: (k+1)*batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % batch_size != 0:
        mini_batch_X = shuffle_X[:, int(m/batch_size)*batch_size:]
        mini_batch_Y = shuffle_Y[:, int(m/batch_size)*batch_size:]
        mini_batches.append((mini_batch_X, mini_batch_Y))
        
    return mini_batches