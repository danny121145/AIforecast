import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def generate_features(df):
    df = df.copy()
    df['Month'] = df['ds'].dt.month
    df['Year'] = df['ds'].dt.year
    df['sin_month'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Lag_1'] = df['y'].shift(1)
    df['Lag_2'] = df['y'].shift(2)
    df['Rolling_Mean_3'] = df['y'].rolling(window=3).mean()
    return df.dropna()

def train_forecast_model(df):
    df = generate_features(df)
    X = df[['Month', 'Year', 'Lag_1', 'Lag_2', 'Rolling_Mean_3', 'sin_month', 'cos_month']]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    return model, mse, rmse, mae, y_test, y_pred, X_test.index

