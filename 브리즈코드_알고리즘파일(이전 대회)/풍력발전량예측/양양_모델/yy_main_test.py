# %%
from . import load_preprocess
def load_preprocessing():
    target = load_preprocess.target
    trains = load_preprocess.trains
    tests = load_preprocess.tests
    return trains, tests, target

# %%
from . import scale_tensor
def split_and_to_tensor(trains, tests):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_trains, X_valids, y_trains, y_valids, X_tests, X_alls = scale_tensor.train_valid_split(trains, tests)
    X_train, X_valid, y_train, y_valid, X_test, X_all = scale_tensor.to_tensor(X_trains, X_valids, y_trains, y_valids, X_tests, X_alls)
    X_train = X_train.to(device)
    X_valid = X_valid.to(device)
    y_train = y_train.to(device)
    y_valid = y_valid.to(device)
    X_test = X_test.to(device)
    X_all = X_all.to(device)
    return X_train, X_valid, y_train, y_valid, X_test, X_all