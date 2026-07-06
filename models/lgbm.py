from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split


def train(X, y, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = LGBMRegressor(random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, X_val, y_val
