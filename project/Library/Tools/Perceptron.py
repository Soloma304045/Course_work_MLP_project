import numpy as np

class Perceptron:
    def __init__(self, input_size, bias=1):
        self.bias = bias
        self.input_size = input_size
        self.weights = np.random.randn(input_size + 1) * 0.1
    
    def forward(self, inputs):
        inputs = np.insert(inputs, 0, self.bias)
        return np.dot(inputs, self.weights)
