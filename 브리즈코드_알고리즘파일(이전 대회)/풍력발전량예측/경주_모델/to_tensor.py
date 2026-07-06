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
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler

def window_df_to_tensor(X_train, y_train, X_valid, y_valid, test, window_size = 31):
    X_train = X_train.copy()
    y_train = y_train.copy()
    X_valid = X_valid.copy()
    y_valid = y_valid.copy()
    test = test.copy()

    scaler = MinMaxScaler()
    tg_scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train.values)
    X_valid_sc = scaler.transform(X_valid.values)
    X_test_sc = scaler.transform(test.values)

    y_train_sc = tg_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_valid_sc = tg_scaler.transform(y_valid.values.reshape(-1, 1))

    X_train_ts = torch.tensor(X_train_sc, dtype = torch.float)
    X_valid_ts = torch.tensor(X_valid_sc, dtype = torch.float)
    X_test_ts = torch.tensor(X_test_sc, dtype = torch.float)
    y_train_ts = torch.tensor(y_train_sc, dtype = torch.float)
    y_valid_ts = torch.tensor(y_valid_sc, dtype = torch.float)

    train_batch_size = len(X_train_ts) - window_size
    valid_batch_size = len(X_valid_ts) - window_size
    test_batch_size = len(X_test_ts) - window_size
    n_features = X_train_ts.shape[-1]

    X_trains = torch.zeros((train_batch_size, window_size, n_features), dtype = torch.float)
    X_valids = torch.zeros((valid_batch_size, window_size, n_features), dtype = torch.float)
    X_tests = torch.zeros((test_batch_size, window_size, n_features), dtype = torch.float)

    y_trains = torch.zeros((train_batch_size, 1), dtype = torch.float)
    y_valids = torch.zeros((valid_batch_size, 1), dtype = torch.float)

    for i in range(train_batch_size):
        X_trains[i] = X_train_ts[i:i+window_size]
        y_trains[i] = y_train_ts[i+window_size]
    
    for i in range(valid_batch_size):
        X_valids[i] = X_valid_ts[i: i+window_size]
        y_valids[i] = y_valid_ts[i+window_size]

    for i in range(test_batch_size):
        X_tests[i] = X_test_ts[i: i+window_size]
    
    return X_trains, X_valids, y_trains, y_valids, X_tests

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
scaler9 = MinMaxScaler()

target_scaler = MinMaxScaler()

scalers = [scaler1, scaler2, scaler3, scaler4, scaler5, scaler6, scaler7, scaler8, scaler9]

def to_tensor(X_train, X_valid, y_train, y_valid, test):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaled_X_trains = {}
    scaled_X_valids = {}
    scaled_X_tests = {}

    X_train_tensors = []
    X_valid_tensors = []
    test_tensors = []

    wtg_names = [f"wtg0{i}"for i in range(1, 10)]
    for scaler in scalers:
        for wtg in wtg_names:
            scaled_X_trains[wtg] = scaler.fit_transform(X_train[wtg])
            scaled_X_valids[wtg] = scaler.transform(X_valid[wtg])
            scaled_X_tests[wtg] = scaler.transform(test[wtg])
    
    scaled_y_train = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
    scaled_y_valid = target_scaler.transform(y_valid.values.reshape(-1, 1))
    
    for wtg in wtg_names:
        X_train_tensor = window_tensor_from_df(scaled_X_trains[wtg], window_size= 24)
        X_train_tensors.append(X_train_tensor)

        X_valid_tensor = window_tensor_from_df(scaled_X_valids[wtg], window_size= 24)
        X_valid_tensors.append(X_valid_tensor)

        test_tensor = window_tensor_from_df(scaled_X_tests[wtg], window_size= 24)
        test_tensors.append(test_tensor)
    
    y_tensor_train = window_tensor_from_df(scaled_y_train, window_size= 24)
    y_tensor_valid = window_tensor_from_df(scaled_y_valid, window_size= 24)

    X_tensor_train = torch.stack(X_train_tensors, dim = 1)
    X_tensor_valid = torch.stack(X_valid_tensors, dim = 1)
    X_tensor_test = torch.stack(test_tensors, dim = 1)

    X_tensor_train = X_tensor_train.to(device)
    X_tensor_valid = X_tensor_valid.to(device)
    y_tensor_train = y_tensor_train.to(device)
    y_tensor_valid = y_tensor_valid.to(device)
    X_tensor_test = X_tensor_test.to(device)

    print(f"X_train tensor shape: {X_tensor_train.shape}")
    print(f"X_valid tensor shape: {X_tensor_valid.shape}")
    print(f"y_train tensor shape: {y_tensor_train.shape}")
    print(f"y_valid tensor shape: {y_tensor_valid.shape}")
    print(f"test_tensor shape: {X_tensor_test.shape}")

    return X_tensor_train, X_tensor_valid, y_tensor_train, y_tensor_valid, X_tensor_test

# %%



