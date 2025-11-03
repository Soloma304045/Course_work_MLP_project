from Library.Tools.Layer import Layer
from Library.Tools.BackPropagation import train_step
import numpy as np

class NeuralNetwork:
    def __init__(self, layers_config, activation_functions):
        self.layers = []
        for i in range(1, len(layers_config)):
            n_inputs = layers_config[i - 1]
            n_neurons = layers_config[i]
            activation = activation_functions[i - 1]
            self.layers.append(Layer(n_inputs, n_neurons, activation))

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, X, y, epochs=1000, learning_rate=0.01, graph=False):
        for epoch in range(epochs):
            total_error = train_step(self, X, y, learning_rate, graph)
            if graph and epoch % 100 == 0:
                print(f"Эпоха {epoch+1}/{epochs}, Ошибка: {total_error:.6f}")