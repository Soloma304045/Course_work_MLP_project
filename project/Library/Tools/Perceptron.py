import numpy as np

class Perceptron:
    def __init__(self, input_size, activate_function, learning_function, bias=1, learning_rate=0.1):
        self.bias = bias
        self.input_size = input_size
        self.weights = np.random.randn(input_size + 1) * 0.1
        self.learning_rate  = learning_rate
        self.function = activate_function
        self.learn_func = learning_function

    def activate(self, z):
        return self.function(z)
        
    def forward(self, inputs):
        if len(inputs) != self.input_size:
            raise ValueError(f"Из {self.input_size} входов получено {len(inputs)}")
        
        inputs = np.insert(inputs, 0, self.bias)
        z = np.dot(inputs, self.weights)

        return self.activate(z)
    
    def train(self, X, y, graph=False):
        return self.learn_func(self, X, y, graph)