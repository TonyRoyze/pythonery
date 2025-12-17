import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, LGBMClassifier

df = pd.read_csv("/Users/danil/Documents/data/house_price.csv")

cols_with_nans = [x for x in df if df[x].isnull().sum() > 0]
df[cols_with_nans].isnull().sum()
num_cols = [col for col in df[cols_with_nans].select_dtypes(exclude="0")]
cat_cols = [col for col in df[cols_with_nans].select_dtypes(include="0")]


def factorize(data):
    for cat_col in data.select_dtypes(include="0"):
        data[cat_col] = pd.factorize(data[cat_col])[0]
    return data


def create_splits(data, col):
    train = data[data["is_nan"] == 0]
    test = data[data["is_nan"] == 1]
    X_train = train.drop([col, "is_nan"], axis=1)
    y_train = train[col]
    X_test = test.drop([col, "is_nan"], axis=1)
    return X_train, y_train, X_test


def train_predict(mode, X_train, y_train, X_test):
    if mode == "regression":
        model = LGBMRegressor()
    else:
        model = LGBMClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred


def impute_missing(df, cols_lst, mode):
    for col in cols_lst:
        if df[col].isnull().sum() > int(len(df) / 2):
            df.drop(col, axis=1, inplace=True)
            print(f"Dropped {col} because it has more than 50% missing values")
        else:
            data = df.copy()
            nan_ixs = np.where(data[col].isna())[0]
            data["is_nan"] = 0
            data.loc[nan_ixs, "is_nan"] = 1
            X = data.drop([col], axis=1)
            y = data[col]
            X = factorize(X)
            data = X.join(y)
            X_train, y_train, X_test = create_splits(data, col)
            y_pred = train_predict(mode, X_train, y_train, X_test)
            df.loc[nan_ixs, col] = y_pred
    return df


df = impute_missing(df, num_cols, "regression")
df = impute_missing(df, cat_cols, "classification")
