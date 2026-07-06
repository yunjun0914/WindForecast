# %%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

def window_tensor_from_df(df, window_size = 24):
  if isinstance(df, np.ndarray):
      data_np = df
  elif isinstance(df, pd.DataFrame):
    df = df.to_numpy()
    data_np = df

  T, F = data_np.shape
  assert T % window_size == 0, f"시계열 길이 {T}는 {window_size}로 나누어 떨어져야 함"
  reshaped = data_np.reshape(-1, window_size, F)
  return torch.tensor(reshaped, dtype = torch.float32)

# %%
from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler()
scaler2 = MinMaxScaler()
scaler3 = MinMaxScaler()
scaler4 = MinMaxScaler()
scaler5 = MinMaxScaler()
scaler6 = MinMaxScaler()
scaler7 = MinMaxScaler()
scaler8 = MinMaxScaler()


scalers = [scaler1, scaler2, scaler3, scaler4, scaler5, scaler6, scaler7, scaler8]

target_scaler = MinMaxScaler()

def train_valid_split(df, test):
    wtg_names = [f"wtg_{i}"for i in range(1, 9)]
    trains = {}
    valids = {}
    tests = {}
    scaled_X_trains = {}
    scaled_X_valids = {}
    scaled_X_tests = {}
    scaled_X_alls = {}
    weather_cols = ['dswrf', 'fvmax_50m', 'fvmin_50m',
               'lhnf', 'maxsa_1p5m', 'mcc', 'mgws_0m', 'p', 'pblh', 'pmsl', 'rh_1p5m',
               'sh_1p5m', 'ta', 'ta_1p5m', 'tdp_1p5m', 'usm_5m', 'uws_10m',
               'vsm_5m', 'vws_10m', 'wind_direction_10m', 'wind_direction_5m', 'wind_strength_10m',
               'wind_strength_5m', "vapor_pressure", "abs_fvmax", "abs_fvmin", "wind_speed_mps", "wind_direction_degree"]

    for wtg in wtg_names:
        df[wtg] = df[wtg].fillna(0)
        trains[wtg] = df[wtg].loc["2024-04-01":"2024-12-01 00:00:00"]
        valids[wtg] = df[wtg].loc["2024-12-01 01:00:00":"2025-02-01 00:00:00"]
        train_target = trains[wtg]["energy_kwh"]
        valid_target = valids[wtg]["energy_kwh"]
        trains[wtg] = trains[wtg][weather_cols]
        valids[wtg] = valids[wtg][weather_cols]
        tests[wtg] = test[wtg][weather_cols]
        df[wtg] = df[wtg][weather_cols]

        for scaler in scalers:
            scaled_X_alls[wtg] = scaler.fit_transform(df[wtg])
            scaled_X_trains[wtg] = scaler.fit_transform(trains[wtg])
            scaled_X_valids[wtg] = scaler.transform(valids[wtg])
            scaled_X_tests[wtg] = scaler.transform(tests[wtg])
        
    scaled_y_train = target_scaler.fit_transform(train_target.values.reshape(-1, 1))
    scaled_y_valid = target_scaler.transform(valid_target.values.reshape(-1, 1))

    return scaled_X_trains, scaled_X_valids, scaled_y_train, scaled_y_valid, scaled_X_tests, scaled_X_alls



# %%
def to_tensor(scaled_X_trains, scaled_X_valids, scaled_y_train, scaled_y_valid, scaled_X_tests, all):
    X_train_tensors = []
    X_valid_tensors = []
    test_tensors = []
    all_X_tensors = []

    wtg_names = [f"wtg_{i}"for i in range(1, 9)]
    
    
    for wtg in wtg_names:
        X_train_tensor = window_tensor_from_df(scaled_X_trains[wtg], window_size= 24)
        X_train_tensors.append(X_train_tensor)

        X_valid_tensor = window_tensor_from_df(scaled_X_valids[wtg], window_size= 24)
        X_valid_tensors.append(X_valid_tensor)

        test_tensor = window_tensor_from_df(scaled_X_tests[wtg], window_size= 24)
        test_tensors.append(test_tensor)

        all_X_tensor = window_tensor_from_df(all[wtg], window_size= 24)
        all_X_tensors.append(all_X_tensor)
    
    y_tensor_train = window_tensor_from_df(scaled_y_train, window_size= 24)
    y_tensor_valid = window_tensor_from_df(scaled_y_valid, window_size= 24)

    X_tensor_train = torch.stack(X_train_tensors, dim = 1)
    X_tensor_valid = torch.stack(X_valid_tensors, dim = 1)
    X_tensor_test = torch.stack(test_tensors, dim = 1)
    X_tensor_all = torch.stack(all_X_tensors, dim = 1)

    print(f"X_train tensor shape: {X_tensor_train.shape}")
    print(f"X_valid tensor shape: {X_tensor_valid.shape}")
    print(f"y_train tensor shape: {y_tensor_train.shape}")
    print(f"y_valid tensor shape: {y_tensor_valid.shape}")
    print(f"test_tensor shape: {X_tensor_test.shape}")
    print(f"all_X_tensor shape: {X_tensor_all.shape}")

    return X_tensor_train, X_tensor_valid, y_tensor_train, y_tensor_valid, X_tensor_test, X_tensor_all


