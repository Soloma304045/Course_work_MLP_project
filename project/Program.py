import numpy as np
import traceback

from Library.Tools.Train_Split import split
from Library.Tools.NeuralNetwork import NeuralNetwork
from Library.Tools.Activate import reLu

try:
    (X_train, X_test), (y_train, y_test), (y_min, y_max) = split(graph=True)

    network = NeuralNetwork(layers_config=[3, 16, 8, 4, 1], activation_functions=[reLu, reLu, reLu, None])
    network.train(X_train, y_train, graph=True)

    y_pred = network.forward(X_test) 
    #y_pred = y_pred * (y_max - y_min) + y_min
    #y_test = y_test * (y_max - y_min) + y_min

    test_mse = np.mean((y_pred - y_test) ** 2)
    print(f"Средняя ошибка на тестовом наборе - {test_mse:.6f}")
    input()

except Exception as e:
    print(f"Произошла ошибка: {e}")
    print("\nПодробный traceback:")
    traceback.print_exc()
    input()