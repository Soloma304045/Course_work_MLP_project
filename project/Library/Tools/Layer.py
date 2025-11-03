import numpy as np

class Layer:
    def __init__(self, n_inputs, n_neurons, activation_function):
        self.weights = np.random.randn(n_inputs + 1, n_neurons) * 0.1
        self.biases = np.random.randn(n_neurons) * 0.1
        self.activation_function = activation_function
        self.output = None

    def forward(self, inputs):
        inputs_with_bias = np.column_stack([np.ones((inputs.shape[0], 1)), inputs])
        
        z = np.dot(inputs_with_bias, self.weights) + self.biases
        if self.activation_function is not None:
            self.output = self.activation_function(z)
        else:
            self.output = z
            
        return self.output