from Layer import Layer

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
