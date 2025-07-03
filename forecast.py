from preprocessing import load_and_prepare
from random_forest_model import train_forecast_model
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("AI Sales Forecaster - Random Forest Edition")

    file_path = input("Enter path to CSV file: ").strip()
    if not os.path.exists(file_path):
        print("\u274C File not found.")
        return

    df_full, all_uids = load_and_prepare(file_path)

    valid_uids = []
    for uid in all_uids:
        df_uid = df_full[df_full['UID'] == uid]
        monthly_points = df_uid.resample('ME', on='Sales Date').sum()
        if len(monthly_points) >= 12:
            valid_uids.append(uid)

    if not valid_uids:
        print("No UIDs found with at least 12 months of data.")
        return

    print(f"Available UIDs with 12+ months of data ({len(valid_uids)}):")
    print(valid_uids)

    uid = input("Enter UID to forecast: ").strip()
    uid = int(uid) if uid.isdigit() else uid
    if uid not in valid_uids:
        print(f"UID {uid} is not valid or lacks sufficient data.")
        return

    try:
        months_ahead = int(input("How many future months to forecast? (3, 6, 9, or 12): ").strip())
        if months_ahead not in [3, 6, 9, 12]:
            raise ValueError()
    except ValueError:
        print("Invalid input. Please enter 3, 6, 9, or 12.")
        return

    df, _ = load_and_prepare(file_path, uid)
    print("Training model and forecasting future sales...")

    model, mape, mse, rmse, mae, y_test, y_pred, test_idx = train_forecast_model(df)

    print("Model Evaluation:")
    print(f"  MSE : {mse:,.2f}")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  MAE : {mae:,.2f}")
    print(f"  MAPE: {mape:.2f}%")

    last_date = df['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months_ahead, freq='MS')
    future_df = pd.DataFrame({'ds': future_dates})
    future_df['Month'] = future_df['ds'].dt.month
    future_df['Year'] = future_df['ds'].dt.year
    future_df['sin_month'] = np.sin(2 * np.pi * future_df['Month'] / 12)
    future_df['cos_month'] = np.cos(2 * np.pi * future_df['Month'] / 12)

    recent = df.copy().tail(6).reset_index(drop=True)
    lags = list(recent['y'].values)[-6:]

    predictions = []
    for i in range(months_ahead):
        lag_1 = lags[-1]
        lag_2 = lags[-2] if len(lags) >= 2 else lag_1
        rolling_3 = np.mean(lags[-3:])

        features = [[
            future_df.loc[i, 'Month'],
            future_df.loc[i, 'Year'],
            lag_1, lag_2, rolling_3,
            future_df.loc[i, 'sin_month'],
            future_df.loc[i, 'cos_month']
        ]]

        feature_names = ['Month', 'Year', 'Lag_1', 'Lag_2', 'Rolling_Mean_3', 'sin_month', 'cos_month']
        input_df = pd.DataFrame(features, columns=feature_names)
        pred = model.predict(input_df)[0]
        predictions.append(pred)
        lags.append(pred)

    future_df['Forecast'] = predictions

    # Save forecast
    forecast_path = f"output/forecast_{uid}.csv"
    os.makedirs("output", exist_ok=True)

    # Actuals + predictions
    df['Prediction'] = np.nan
    df.loc[test_idx, 'Prediction'] = y_pred

    combined_df = pd.concat([
        df[['ds', 'y', 'Prediction']],
        future_df[['ds', 'Forecast']].assign(y=np.nan, Prediction=np.nan)
    ], ignore_index=True)

    # Round values
    combined_df['y'] = combined_df['y'].round(2)
    combined_df['Prediction'] = combined_df['Prediction'].round(2)
    combined_df['Forecast'] = combined_df['Forecast'].round(2)

    # Drop rows with no prediction or forecast
    final_output = combined_df.dropna(subset=['Prediction', 'Forecast'], how='all')

    # Save to CSV
    final_output.to_csv(forecast_path, index=False)
    print(f"Cleaned forecast saved to {forecast_path}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(combined_df['ds'], combined_df['y'], label='Actual', marker='o')
    plt.plot(combined_df['ds'], combined_df['Prediction'], label='Prediction', linestyle='--')
    plt.plot(combined_df['ds'], combined_df['Forecast'], label='Forecast', linestyle='dotted', marker='x')
    plt.title(f"Sales Forecast for UID {uid}")
    plt.xlabel("Date")
    plt.ylabel("Amount (USD)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()