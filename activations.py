
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def identity(x):
    return x

def sin(x):
    return np.sin(x)

def relu(x):
    return np.maximum(0, x)

def cos(x):
    return np.cos(x)

def gaussian(x):
    return np.exp(-x**2)

def abs(x):
    return np.abs(-x**2)

def square(x):
    return x**2

def step(x):
    return np.where(x > 0, 1, 0)

def string_to_fn(string):
    if string == 'sigmoid':
        return sigmoid
    elif string == 'tanh':
        return tanh
    elif string == 'identity':
        return identity
    elif string == 'sin':
        return sin
    elif string == 'relu':
        return relu
    elif string == 'cos':
        return cos
    elif string == 'gaussian':
        return gaussian
    elif string == 'square':
        return square
    elif string == 'step':
        return step
    else:
        raise ValueError('Unknown activation function: {}'.format(string))
    
def fn_to_string(fn):
    if fn == sigmoid:
        return 'sigmoid'
    elif fn == tanh:
        return 'tanh'
    elif fn == identity:
        return 'identity'
    elif fn == sin:
        return 'sin'
    elif fn == relu:
        return 'relu'
    elif fn == cos:
        return 'cos'
    elif fn == gaussian:
        return 'gaussian'
    elif fn == square:
        return 'square'
    elif fn == step:
        return 'step'
    else:
        raise ValueError('Unknown activation function: {}'.format(fn))