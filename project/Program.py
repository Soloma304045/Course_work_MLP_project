import pandas as pd
import traceback

from Library.Tools.Train_Split import split_simple
from Library.Tools.NeuralNetwork import NeuralNetwork
from Library.Tools.Sigmoid import sigmoid

try:
    (X_train, X_test), (y_train, y_test) = split_simple()

    network = NeuralNetwork(layers_config=[3, 16, 8, 1], activation_functions=[sigmoid, sigmoid, None])
    network.train(X_train, y_train, graph=True)
    input()

except Exception as e:
    print(f"Произошла ошибка: {e}")
    print("\nПодробный traceback:")
    traceback.print_exc()
    input()