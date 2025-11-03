import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons, activation_function):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1   
        self.biases = np.zeros((1, n_neurons))                      
        self.activation_function = activation_function
        self.output = None
        self.last_input = None
        self.z = None

    def forward(self, inputs):
        self.last_input = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        if self.activation_function is None:
            self.output = self.z
        else:
            self.output = self.activation_function(self.z)
        return self.output
