import numpy as np

# Gradient Descent

def update_parameters(parameters, grads, learning_rate=0.01):
    L = len(parameters) // 2
    
    for l in range(L):
        parameters['W' + str(l+1)] -= learning_rate * grads['dW' + str(l+1)]
        parameters['b' + str(l+1)] -= learning_rate * grads['db' + str(l+1)]
    return parameters


# momentum
def initialize_velocity(parameters):
    L = len(parameters)//2
    v = {}
    for l in range(1, L+1):
        v['dW' + str(l)] = np.zeros((parameters['W' + str(l)].shape[0], parameters['W' + str(l)].shape[1]))
        v['db' + str(l)] = np.zeros((parameters['b' + str(l)].shape[0], parameters['b' + str(l)].shape[1]))
    return v


def update_parameters_with_momentum(parameters, grads, v, beta=0.9, learning_rate=1e-3):
    L = len(parameters) // 2
    for l in range(1, L+1):
        v['dW' + str(l)] = beta * v['dW' + str(l)] + (1 - beta) * grads['dW' + str(l)]
        v['db' + str(l)] = beta * v['db' + str(l)] + (1 - beta) * grads['db' + str(l)]
        
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * v['dW' + str(l)]
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * v['db' + str(l)]
    
    return parameters, v


# adam
def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    for l in range(1, L+1):
        v['dW' + str(l)] = np.zeros((parameters['W' + str(l)].shape[0], parameters['W' + str(l)].shape[1]))
        v['db' + str(l)] = np.zeros((parameters['b' + str(l)].shape[0], parameters['b' + str(l)].shape[1]))
        
        s['dW' + str(l)] = np.zeros((parameters['W' + str(l)].shape[0], parameters['W' + str(l)].shape[1]))
        s['db' + str(l)] = np.zeros((parameters['W' + str(l)].shape[0], parameters['b' + str(l)].shape[1]))
    return v, s


def update_parameters_with_adam(parameters, v, s, t, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    v_correct = {}
    s_correct = {}
    
    for l in range(1, L+1):
        v['dW' + str(l)] = beta1 * v['dW' + str(l)] + (1-beta1) * v['dW' + str(l)]
        v['db' + str(l)] = beta1 * v['db' + str(l)] + (1-beta1) * v['db' + str(l)]
        
        v_correct['dW' + str(l)] = v['dW' + str(l)] / (1.0 * np.power(beta1, t))
        v_correct['db' + str(l)] = v['db' + str(l)] / (1.0 * np.power(beta1, t))
        
        s['dW' + str(l)] = beta2 * s['dW' + str(l)] + (1-beta2) * v['dW' + str(l)]
        s['db' + str(l)] = beta2 * s['db' + str(l)] + (1-beta2) * v['db' + str(l)]
        
        s_correct['dW' + str(l)] = s['dW' + str(l)] / (1.0 - np.power(beta2, t))
        s_correct['db' + str(l)] = s['db' + str(l)] / (1.0 - np.power(beta2, t)) 
        
        parameters['W' + str(l)] -= learning_rate * v['dW' + str(l)] / np.sqrt(s['dW' + str(l)] + epsilon)
        parameters['b' + str(l)] -= learning_rate * v['db' + str(l)] / np.sqrt(s['db' + str(l)] + epsilon)
        
    return parameters, v, s, v_correct, s_correct
    