import numpy as np


def sigmoid(Z):
    cache = Z
    s = 1.0 / (1.0 + np.exp(-Z))
    return s, cache

def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dSig = dA * s * (1 - s)
    return dSig


def relu(Z):
    cache = Z
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape)
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    
    dZ[Z <= 0] = 0
    # dZ[Z > 0] = 1
    assert(dZ.shape == Z.shape)
    return dZ

def tanh(Z):
    cache  = Z
    tn =  1 - (2 / (np.exp(2 * Z) + 1))
    return tn, cache

def tanh_backward(dA, cache):   
    Z = cache
    tanh = 1 - (2 / (np.exp(2*Z) + 1))
    dTanh = dA * (1 - tanh * tanh)
    
    