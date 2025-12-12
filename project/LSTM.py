import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import traceback
import warnings
warnings.filterwarnings("ignore")


from Library.Tools.Train_Split import split_lstm
from Library.Tools.LSTM_Simple import SimpleLSTM


def train_lstm(model, X_train, y_train, epochs=200, lr=0.001, graph=True):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)

        loss.backward()
        optimizer.step()

        if graph and epoch % 10 == 0:
            print(f"Эпоха {epoch}/{epochs}, Ошибка: {loss.item():.6f}")

    return model


def evaluate_lstm(model, X_test, y_test):
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        y_pred = model(X_tensor)
        mse = torch.mean((y_pred - y_tensor) ** 2).item()

    return mse


if __name__ == "__main__":
    try:
        print("=== Загрузка и подготовка данных ===")
        (X_train, X_test), (y_train, y_test), (y_min, y_max) = split_lstm(
            sequence_length=10,
            graph=True
        )

        print("\n=== Инициализация модели LSTM ===")
        model = SimpleLSTM(
            input_size=3,
            hidden_size=16, 
            num_layers=2,   
            output_size=1,
            dropout=0.1
        )

        print("\n=== Обучение модели ===")
        model = train_lstm(
            model,
            X_train,
            y_train,
            epochs=100,
            lr=0.1,
            graph=True
        )

        print("\n=== Тестирование модели ===")
        mse = evaluate_lstm(model, X_test, y_test)
        print(f"Средняя ошибка на тестовом наборе (LSTM): {mse:.6f}")

        input("\nНажмите Enter для выхода...")

    except Exception as e:
        print(f"Произошла ошибка: {e}")
        print("\nПодробный traceback:")
        traceback.print_exc()
        input()
