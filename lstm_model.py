import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        targets.append(data[i+seq_length])
    return np.array(sequences), np.array(targets)

def train_lstm_model(df, seq_length=6, epochs=100, lr=0.001):
    df = df.copy()
    scaler = MinMaxScaler()
    df['y_scaled'] = scaler.fit_transform(df[['y']])

    data = df['y_scaled'].values
    X_all, y_all = create_sequences(data, seq_length)

    # Split into train and test sets (e.g., 80% train, 20% test)
    split_index = int(len(X_all) * 0.8)
    X_train, X_test = X_all[:split_index], X_all[split_index:]
    y_train, y_test = y_all[:split_index], y_all[split_index:]

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)

    y_pred = scaler.inverse_transform(y_pred_tensor.numpy())
    y_test_unscaled = scaler.inverse_transform(y_test_tensor.numpy())

    mse = mean_squared_error(y_test_unscaled, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_unscaled, y_pred)
    nonzero = y_test_unscaled != 0
    mape = np.mean(np.abs((y_test_unscaled[nonzero] - y_pred[nonzero]) / (y_test_unscaled[nonzero]))) * 100 

    last_seq = torch.tensor(X_all[-1:], dtype=torch.float32).unsqueeze(-1)

    return model, scaler, mse, rmse, mae, mape, y_test_unscaled, y_pred, last_seq

def forecast_lstm(model, last_seq, scaler, future_months, last_date):
    preds_scaled = []
    model.eval()
    seq = last_seq.clone()

    for _ in range(future_months):
        with torch.no_grad():
            pred = model(seq).item()
        preds_scaled.append(pred)
        # Add new prediction to sequence
        next_input = torch.tensor([[[pred]]], dtype=torch.float32)
        seq = torch.cat((seq[:, 1:, :], next_input), dim=1)

    preds_unscaled = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_months, freq='MS')
    future_df = pd.DataFrame({'ds': future_dates, 'Forecast': np.round(preds_unscaled, 2)})
    return future_df
