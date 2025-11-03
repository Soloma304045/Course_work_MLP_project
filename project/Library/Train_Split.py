import pandas as pd
import numpy as np
import os

def split():

    file_path = os.path.join(r'C:\Users\Иван\Desktop\MLP\project\Library\data', 'T1.csv')    
    df = pd.read_csv(file_path, sep=',')

    print(df.head())

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
    print(df.head())


    X = df.drop(columns=['LV ActivePower (kW)'])
    y = df['LV ActivePower (kW)']
    split_index = int(len(df) * 0.8)
    shuffled = np.random.permutation(len(df))

    train_index = shuffled[:split_index]
    test_index = shuffled[split_index:]

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    return (X_train, X_test), (y_train, y_test)