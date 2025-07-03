from preprocessing import load_and_prepare
from random_forest_model import train_forecast_model
from lstm_model import train_lstm_model, forecast_lstm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("\U0001F4CA AI Sales Forecaster")

    file_path = input("Enter path to CSV file: ").strip()
    if not os.path.exists(file_path):
        print("❌ File not found.")
        return

    df_full, all_uids = load_and_prepare(file_path)

    valid_uids = []
    for uid in all_uids:
        df_uid = df_full[df_full['UID'] == uid]
        monthly_points = df_uid.resample('ME', on='Sales Date').sum()
        if len(monthly_points) >= 12:
            valid_uids.append(uid)

    if not valid_uids:
        print("⚠️ No UIDs found with at least 12 months of data.")
        return

    print(f"\n✅ Available UIDs with 12+ months of data ({len(valid_uids)}):")
    print(valid_uids)

    uid = input("\nEnter UID to forecast: ").strip()
    if uid.isdigit():
        uid = int(uid)

    if uid not in valid_uids:
        print(f"❌ UID {uid} is not valid or lacks sufficient data.")
        return

    try:
        months_ahead = int(input("How many future months to forecast? (3, 6, 9, or 12): ").strip())
        if months_ahead not in [3, 6, 9, 12]:
            raise ValueError()
    except ValueError:
        print("❌ Invalid input. Please enter 3, 6, 9, or 12.")
        return

    model_choice = input("\nSelect model type (rf = Random Forest, lstm = LSTM): ").strip().lower()

    df, _ = load_and_prepare(file_path, uid)
    history_df = df[['ds', 'y']].copy()

    print("\n🔁 Training model and forecasting future sales...")

    if model_choice == 'rf':
        model, mape, mse, rmse, mae, y_test, y_pred, test_idx = train_forecast_model(df)

        print("\n✅ Model Evaluation:")
        print(f"  MSE : {mse:,.2f}")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  MAE : {mae:,.2f}")
        print(f"  MAPE: {mape:.2f}%")

        df['Prediction'] = np.nan
        df.loc[test_idx, 'Prediction'] = y_pred

        # Forecast future months (reuse code from RF version)
        last_date = df['ds'].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months_ahead, freq='MS')
        future_df = pd.DataFrame({'ds': future_dates})
        future_df['Month'] = future_df['ds'].dt.month
        future_df['Year'] = future_df['ds'].dt.year
        future_df['Quarter'] = future_df['ds'].dt.quarter

        recent = df.copy().tail(6).reset_index(drop=True)
        lags = list(recent['y'].values)[-6:]

        predictions = []
        for i in range(len(future_df)):
            lag_1 = lags[-1]
            lag_2 = lags[-2] if len(lags) >= 2 else lag_1
            rolling_3 = np.mean(lags[-3:]) if len(lags) >= 3 else lag_1

            feature_row = [[
                future_df.loc[i, 'Month'],
                future_df.loc[i, 'Year'],
                lag_1, lag_2, rolling_3
            ]]
            feature_names = ['Month', 'Year', 'Lag_1', 'Lag_2', 'Rolling_Mean_3']
            input_df = pd.DataFrame(np.array(feature_row).reshape(1, -1), columns=feature_names)
            pred = model.predict(input_df)[0]
            predictions.append(round(pred, 2))
            lags.append(pred)

        future_df['Forecast'] = predictions

    elif model_choice == 'lstm':
        model, scaler, mse, rmse, mae, mape, y_test, y_pred, last_seq = train_lstm_model(df)

        print("\n✅ Model Evaluation:")
        print(f"  MSE : {mse:,.2f}")
        print(f"  RMSE: {rmse:,.2f}")
        print(f"  MAE : {mae:,.2f}")
        print(f"  MAPE: {mape:.2f}%")

        df['Prediction'] = np.nan
        df.iloc[-len(y_pred):, df.columns.get_loc('Prediction')] = y_pred

        future_df = forecast_lstm(model, last_seq, scaler, months_ahead, df['ds'].max())


    else:
        print("❌ Invalid model selection.")
        return

    # Combine + Clean
    future_df['y'] = np.nan
    future_df['Prediction'] = np.nan
    combined_df = pd.concat([df[['ds', 'y', 'Prediction']], future_df[['ds', 'y', 'Prediction', 'Forecast']]], ignore_index=True)
    combined_df.dropna(subset=['Prediction', 'Forecast'], how='all', inplace=True)

    forecast_path = f"output/forecast_{uid}.csv"
    os.makedirs("output", exist_ok=True)
    combined_df.to_csv(forecast_path, index=False)
    print(f"\n📁 Cleaned forecast saved to {forecast_path}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(combined_df['ds'], combined_df['y'], label='Actual', marker='o')
    plt.plot(combined_df['ds'], combined_df['Prediction'], label='Prediction', marker='x', linestyle='--')
    plt.plot(combined_df['ds'], combined_df['Forecast'], label='Forecast', marker='^', linestyle=':')
    plt.title(f"Sales Forecast for UID {uid}")
    plt.xlabel("Date")
    plt.ylabel("Amount (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()