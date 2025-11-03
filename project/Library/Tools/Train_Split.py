import pandas as pd
import numpy as np
import os

def split(test_size=0.2, random_state=42, graph=False):
    data_path = os.path.join(r'C:\Users\Иван\Desktop\MLP\project\Library\data', 'T1.csv')
    df = pd.read_csv(data_path, sep=',')

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
    
    if graph:
        print(f"\nФормы перед батчингом:")
        print(f"X: {X.shape}, y: {y.shape}")
        print(f"Колонки X: {list(X.columns)}")
    
    batches = len(X) // 6
    records = batches * 6
    
    if graph:
        print(f"\nГруппировка данных:")
        print(f"Всего записей: {len(X)}")
        print(f"Количество полных батчей: {batches}")
        print(f"Используется записей: {records}")
        print(f"Отброшено записей: {len(X) - records}")
    
    X_trimmed = X.iloc[:records].reset_index(drop=True)
    y_trimmed = y.iloc[:records].reset_index(drop=True)
    
    X_batches = []
    y_batches = []
    
    for i in range(batches):
        start_idx = i * 6
        end_idx = start_idx + 6
        
        X_batch = X_trimmed.iloc[start_idx:end_idx]
        y_batch = y_trimmed.iloc[start_idx:end_idx]

        X_flat = X_batch.values.flatten()
        y_value = y_batch.iloc[-1]
        
        X_batches.append(X_flat)
        y_batches.append(y_value)
    
    X_batch = np.array(X_batches)
    y_batch = np.array(y_batches)
    
    print(f"\nФорма X_batches: {X_batch.shape}")
    print(f"Форма y_batches: {y_batch.shape}")
    
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(batches)
    
    X_shuffled = X_batch[shuffled_indices]
    y_shuffled = y_batch[shuffled_indices]
    
    split_index = int(batches * (1 - test_size))
    
    X_train = X_shuffled[:split_index]
    X_test = X_shuffled[split_index:]
    y_train = y_shuffled[:split_index]
    y_test = y_shuffled[split_index:]
    
    print(f"\nИтоговые размеры выборок:")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
    print(f"Количество train батчей: {len(X_train)}")
    print(f"Количество test батчей: {len(X_test)}")
    
    return (X_train, X_test), (y_train, y_test)

# Альтернативная версия без батчинга (если вышеописанная не работает)
def split_simple(test_size=0.2, random_state=42, data_path=None):
    """
    Упрощенная версия без батчинга для проверки работы нейронной сети
    """
    # Путь к данным
    if data_path is None:
        data_path = os.path.join(r'C:\Users\Иван\Desktop\MLP\project\Library\data', 'T1.csv')
    
    # Загрузка данных
    df = pd.read_csv(data_path, sep=',')
    
    # Предобработка данных (та же самая)
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
    
    columns_to_drop = ['Date/Time', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']
    df_processed = df_normalized.drop(columns=columns_to_drop)
    
    # Разделение на признаки и целевую переменную
    X = df_processed.drop(columns=['LV ActivePower (kW)'])
    y = df_processed['LV ActivePower (kW)']
    
    # Обычное разделение без батчинга
    split_index = int(len(X) * 0.8)
    shuffled = np.random.permutation(len(X))
    
    train_index = shuffled[:split_index]
    test_index = shuffled[split_index:]
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Преобразуем в numpy arrays
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    
    print(f"Упрощенное разделение:")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
    
    return (X_train, X_test), (y_train, y_test)