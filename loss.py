import numpy as np

def binary_cross_entropy_loss(AL, Y):
    m = Y.shape[1]
    
    cost = (-1.0 / m) * np.sum((Y * np.log(AL)) + ((1 - Y) * np.log(1 - AL)))
    
    assert(cost.shape == ())
    return cost
    
def cross_entropy_loss(AL, Y):
    m = Y.shape[1]
    
    cost = (-1 / m) * np.sum(Y * np.log(AL))
    
    assert(cost.shape == ())
    return cost