import numpy as np
from activation_functions import *
import h5py


def linear(X, W, b):
    Z = np.dot(W, X) + b
    cache = (X, W, b)
    return Z, cache

def linear_activation(A_prev, W, b, activation_name):
    if activation_name.lower() == 'sigmoid':
        Z, linear_cache = linear(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation_name.lower() == 'relu':
        Z, linear_cache = linear(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)
    
    return A, cache   

def forward(X, parameters):
    caches = list()
    A = X
    L = len(parameters)//2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation(A, parameters['W' + str(l)], parameters['b' + str(l)], activation_name='relu')
        caches.append(cache)
    
    AL, cache = linear_activation(A, parameters['W' + str(L)], parameters['b' + str(L)], activation_name='sigmoid')
    caches.append(cache)
    return AL, caches

def forward_dropout(X, parameters, keep_prob, layer_start, layer_end):
    np.random.seed(3)
    caches = list()
    A = X
    L = len(parameters)//2
    
    for l in range(1, L):
        A_prev = A
        if l == layer_start and l <= layer_end:
            A, cache = linear_activation(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation_name='relu')
            D = np.random.randn(A.shape[0], A.shape[1])
            D = (D < keep_prob).astype(int)
            A = (A * D) / keep_prob 
        else:
            A, cache = linear_activation(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation_name='relu')
        caches.append(cache)
    AL, cache = linear_activation(A, parameters['W' + str(L)], parameters['b' + str(L)], activation_name='sigmoid')
    caches.append(cache)
    
    assert(AL.shape == (1, X.shape[1]))
    
    return AL, caches
        
