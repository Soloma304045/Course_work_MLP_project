import pandas as pd
import numpy as np
import torch
import os

def split(test_size=0.2, random_state=42, graph=False):
    data_path = os.path.join(r'C:\Users\Иван\Desktop\MLP\project\Library\data', 'T1.csv')
    df = pd.read_csv(data_path, sep=',')

    df = df[df['LV ActivePower (kW)'] >= 0]
    df = df[~((df['Wind Speed (m/s)'] > 2) & (df['LV ActivePower (kW)'] == 0))]

    if graph:
        print("Первые 5 строк данных:")
        print(df.head())
        print(f"Всего записей: {len(df)}")

    df_normalized = df.copy()
    wind_min = df_normalized['Wind Speed (m/s)'].min()
    wind_max = df_normalized['Wind Speed (m/s)'].max()
    df_normalized['Wind Speed (m/s)'] = (df_normalized['Wind Speed (m/s)'] - wind_min) / (wind_max - wind_min)

    df_normalized['Wind Direction_sin'] = np.sin(2 * np.pi * df['Wind Direction (°)'] / 360)
    df_normalized['Wind Direction_cos'] = np.cos(2 * np.pi * df['Wind Direction (°)'] / 360)

    sin_min = df_normalized['Wind Direction_sin'].min()
    sin_max = df_normalized['Wind Direction_sin'].max()
    cos_min = df_normalized['Wind Direction_cos'].min()
    cos_max = df_normalized['Wind Direction_cos'].max()

    df_normalized['Wind Direction_sin'] = (df_normalized['Wind Direction_sin'] - sin_min) / (sin_max - sin_min)
    df_normalized['Wind Direction_cos'] = (df_normalized['Wind Direction_cos'] - cos_min) / (cos_max - cos_min)

    df = df_normalized.drop(columns=['Date/Time', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)'])

    

    if graph:
        print("\nДанные после предобработки:")
        print(df.head())
        print(f"Форма данных после предобработки: {df.shape}")

    X = df.drop(columns=['LV ActivePower (kW)'])
    y = df['LV ActivePower (kW)']

    y_min = y.min()
    y_max = y.max()
    y = (y - y_min) / (y_max - y_min)

    if graph:
        print(f"\nФормы перед батчингом:")
        print(f"X: {X.shape}, y: {y.shape}")
        print(f"Колонки X: {list(X.columns)}")

    batch_size = 6
    batches = len(X) // batch_size
    records = batches * batch_size

    if graph:
        print(f"\nГруппировка данных:")
        print(f"Всего записей: {len(X)}")
        print(f"Количество полных батчей: {batches}")
        print(f"Используется записей: {records}")
        print(f"Отброшено записей: {len(X) - records}")

    X_final = X.iloc[:records].reset_index(drop=True)
    y_final = y.iloc[:records].reset_index(drop=True)

    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(batches)

    X_shuffled = X_final.iloc[shuffled_indices]
    y_shuffled = y_final.iloc[shuffled_indices]

    split_index = int(batches * (1 - test_size))

    X_train = X_shuffled[:split_index]
    X_test = X_shuffled[split_index:]
    y_train = y_shuffled[:split_index]
    y_test = y_shuffled[split_index:]

    feature_names = list(X.columns)
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    y_train = pd.DataFrame(y_train, columns=['LV ActivePower (kW)'])
    y_test = pd.DataFrame(y_test, columns=['LV ActivePower (kW)'])

    if graph:
        print(f"\nИтоговые размеры выборок:")
        print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
        print(f"Количество train записей: {len(X_train)}")
        print(f"Количество test записей: {len(X_test)}")

    return (X_train, X_test), (y_train, y_test), (y_min, y_max)

def split_lstm(sequence_length=10, test_size=0.2, graph=False):
    data_path = os.path.join(
        r'C:\Users\Иван\Desktop\MLP\project\Library\data', 'T1.csv'
    )
    df = pd.read_csv(data_path, sep=',')

    df = df[df['LV ActivePower (kW)'] >= 0]
    df = df[~((df['Wind Speed (m/s)'] > 2) & (df['LV ActivePower (kW)'] == 0))]

    if graph:
        print("\nПервые строки исходных данных:")
        print(df.head())
        print(f"Всего записей: {len(df)}")

    df_norm = df.copy()

    w_min = df_norm['Wind Speed (m/s)'].min()
    w_max = df_norm['Wind Speed (m/s)'].max()
    df_norm['Wind Speed (m/s)'] = (df_norm['Wind Speed (m/s)'] - w_min) / (w_max - w_min)

    df_norm['Wind Direction_sin'] = np.sin(2 * np.pi * df['Wind Direction (°)'] / 360)
    df_norm['Wind Direction_cos'] = np.cos(2 * np.pi * df['Wind Direction (°)'] / 360)

    sin_min = df_norm['Wind Direction_sin'].min()
    sin_max = df_norm['Wind Direction_sin'].max()
    cos_min = df_norm['Wind Direction_cos'].min()
    cos_max = df_norm['Wind Direction_cos'].max()

    df_norm['Wind Direction_sin'] = (df_norm['Wind Direction_sin'] - sin_min) / (sin_max - sin_min)
    df_norm['Wind Direction_cos'] = (df_norm['Wind Direction_cos'] - cos_min) / (cos_max - cos_min)

    df_norm = df_norm.drop(columns=[
        'Date/Time',
        'Theoretical_Power_Curve (KWh)',
        'Wind Direction (°)'
    ])

    if graph:
        print("\nПосле нормализации:")
        print(df_norm.head())
        print(df_norm.shape)

    X_all = df_norm[['Wind Speed (m/s)', 'Wind Direction_sin', 'Wind Direction_cos']]
    y_all = df_norm['LV ActivePower (kW)']

    y_min = y_all.min()
    y_max = y_all.max()
    y_all = (y_all - y_min) / (y_max - y_min)

    X_list = []
    y_list = []

    for i in range(len(X_all) - sequence_length):
        seq = X_all.iloc[i:i + sequence_length].values
        target = y_all.iloc[i + sequence_length]

        X_list.append(seq)
        y_list.append(target)

    X_np = np.array(X_list, dtype=np.float32)
    y_np = np.array(y_list, dtype=np.float32).reshape(-1, 1)

    if graph:
        print("\nФормы LSTM-датасета:")
        print("X_np:", X_np.shape)
        print("y_np:", y_np.shape)

    split_index = int(len(X_np) * (1 - test_size))

    X_train = X_np[:split_index]
    X_test = X_np[split_index:]

    y_train = y_np[:split_index]
    y_test = y_np[split_index:]

    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    if graph:
        print("\nИтоговые формы тензоров:")
        print("X_train:", X_train.shape)
        print("X_test:", X_test.shape)
        print("y_train:", y_train.shape)
        print("y_test:", y_test.shape)

    return (X_train, X_test), (y_train, y_test), (y_min, y_max)
