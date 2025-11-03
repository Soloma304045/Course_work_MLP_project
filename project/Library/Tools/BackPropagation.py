import numpy as np
from Library.Tools.Sigmoid import sigmoid_derivative

def train_step(network, X, y, learning_rate, graph):
    outputs = [X]
    for layer in network.layers:
        X = layer.forward(X)
        outputs.append(X)
    
    # Вычисление ошибки
    error = outputs[-1] - y.reshape(-1, 1)
    total_error = np.mean(np.abs(error))
    
    # Обратное распространение
    for i in reversed(range(len(network.layers))):
        layer = network.layers[i]
        layer_output = outputs[i + 1]
        layer_input = outputs[i]
        
        # Вычисление дельты
        if i == len(network.layers) - 1:  # Выходной слой
            delta = error
        else:  # Скрытые слои
            next_layer = network.layers[i + 1]
            # Суммируем вклады в ошибку от следующего слоя
            error_back = np.dot(delta, next_layer.weights.T)
            if layer.activation_function != None:
                delta = error_back * sigmoid_derivative(layer_output)
            else:
                delta = error_back
        
        # Обновление весов и смещений
        # Добавляем смещение к входным данным
        layer_input_with_bias = np.column_stack([np.ones((layer_input.shape[0], 1)), layer_input])
        
        # Вычисляем градиенты
        dw = np.dot(layer_input_with_bias.T, delta) / X.shape[0]
        db = np.mean(delta, axis=0)
        
        # Обновляем параметры
        layer.weights -= learning_rate * dw
        layer.biases -= learning_rate * db
    
    return total_error