import numpy as np
from project.Library.Tools.Activate import reLu_derivative

def train_step(network, X, y, learning_rate, graph):
    outputs = [X]
    for layer in network.layers:
        X = layer.forward(X)
        outputs.append(X)
    
    error = outputs[-1] - y.values.reshape(-1, 1)
    mse = np.mean(error ** 2)  

    for i in reversed(range(len(network.layers))):
        layer = network.layers[i]
        layer_output = outputs[i + 1]
        layer_input = outputs[i]
        
        if i == len(network.layers) - 1:
            delta = error
        else:
            next_layer = network.layers[i + 1]
            error_back = np.dot(delta, next_layer.weights.T)
            delta = error_back * reLu_derivative(layer_output)
        
        layer_inputs = np.column_stack([np.ones((layer_input.shape[0], 1)), layer_input])
        
        dw_combined = np.dot(layer_inputs.T, delta) / X.shape[0]
        dw_combined = np.clip(dw_combined, -1.0, 1.0)

        db = dw_combined[0].reshape(1, -1)
        dw = dw_combined[1:]
        
        layer.weights -= learning_rate * dw
        layer.biases -= learning_rate * db
    
    return mse