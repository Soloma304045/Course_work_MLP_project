import numpy as np

from Perceptron import Perceptron

class Layer:
    def __init__(self, n_inputs, n_neurons, activation_function):
        self.neurons = [Perceptron(n_inputs) for _ in range(n_neurons)]
        self.activation_function = activation_function
        self.output = None

    def forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            z = neuron.forward(inputs)
            outputs.append(z)
        outputs = np.array(outputs)
        self.output = self.activation_function(outputs)
        return self.output

