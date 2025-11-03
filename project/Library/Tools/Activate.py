import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(s):
    return s * (1 - s)

def reLu(x):
    return np.maximum(0, x)

def reLu_derivative(x):
    return (x > 0).astype(float)