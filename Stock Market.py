# STEP 1 - Installing Necessary Libraries

!pip install pmdarima
!pip install prophet
!pip install keras
!pip install yfinance --upgrade --no-cache-dir

# STEP 2 - Importing & Cleaning the Dataset

from google.colab import files
import pandas as pd

# Upload the dataset
uploaded = files.upload()

# Load and clean the dataset
df = pd.read_csv("Yahoo Finance Stock Data.csv")

# Convert date with UTC handling to avoid timezone warnings
df.columns = df.columns.str.strip()
df["Date"] = pd.to_datetime(df["Date"], utc=True)
df = df.sort_values(by=["Company", "Date"]).reset_index(drop=True)

# Save cleaned dataset for further use
df.to_csv("Cleaned Yahoo Finance Stock Data.csv", index=False)
df.head()

# STEP 3 - Visualizing the Data

import matplotlib.pyplot as plt

# Limit to first 10 companies only
companies = df["Company"].unique()[:50]

for company in companies:
    company_df = df[df["Company"] == company]
    plt.figure(figsize=(10, 4))
    plt.plot(company_df["Date"], company_df["Close"])
    plt.title(f"{company} - Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.grid()
    plt.show()

# STEP 4 - ARIMA Model Forecasting

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

results = []

# Limit to first 10 companies for demonstration
for company in companies[:50]:
    company_df = df[df["Company"] == company].copy()
    company_df["Date"] = pd.to_datetime(company_df["Date"])
    company_df = company_df.sort_values("Date")
    ts = company_df["Close"].values

    split = int(0.8 * len(ts))
    train, test = ts[:split], ts[split:]

    # Skip very short series
    if len(train) < 30 or len(test) < 5:
        print(f"Skipping {company} due to insufficient data")
        continue

    try:
        # Fit ARIMA model (simple order; you can loop to find best (p,d,q))
        model = ARIMA(train, order=(5, 1, 0))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=len(test))
        mse = mean_squared_error(test, forecast)
        results.append((company, "ARIMA", mse))

        # Plotting
        plt.figure(figsize=(10, 4))
        plt.plot(company_df["Date"][split:], test, label="Actual")
        plt.plot(company_df["Date"][split:], forecast, label="Forecast", color="red")
        plt.title(f"{company} - ARIMA Forecast")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid()
        plt.show()

    except Exception as e:
        print(f"Skipping {company} due to ARIMA fitting issue: {e}")

# STEP 5 - SARIMA Model Forecasting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

results_sarima_statsmodels = []

# Limit to 10 companies for visualization
companies = df["Company"].unique()[:50]

for company in companies:
    company_df = df[df["Company"] == company].copy()
    company_df = company_df.sort_values("Date")
    ts = company_df["Close"].values

    split = int(0.8 * len(ts))
    train, test = ts[:split], ts[split:]

    try:
        # SARIMA model: order=(p,d,q), seasonal_order=(P,D,Q,s)
        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12), enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=len(test))
        mse = mean_squared_error(test, forecast)
        results_sarima_statsmodels.append((company, "SARIMA", mse))

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(company_df["Date"][split:], test, label="Actual")
        plt.plot(company_df["Date"][split:], forecast, label="Forecast", color="green")
        plt.title(f"{company} - SARIMA Forecast (statsmodels)")
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.legend()
        plt.grid()
        plt.show()

    except Exception as e:
        print(f"Skipping {company} due to SARIMA (statsmodels) error: {e}")

# STEP 6 - Prophet Model Forecasting

import logging
import prophet
logger = logging.getLogger("cmdstanpy")
logger.setLevel(logging.WARNING)

from prophet import Prophet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

results_prophet = []

for company in companies[:50]:  # Limit for testing
    try:
        # Step 1: Subset and clean company data
        company_df = df[df["Company"] == company].copy()
        company_df["Date"] = pd.to_datetime(company_df["Date"]).dt.tz_localize(None)  # Remove timezone
        company_df = company_df.sort_values("Date")

        # Step 2: Prepare DataFrame for Prophet
        prophet_df = company_df[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        prophet_df = prophet_df.set_index("ds").resample("D").mean()  # Fill missing days with NaN
        prophet_df["y"] = prophet_df["y"].ffill()  # Forward fill missing prices
        prophet_df = prophet_df.reset_index()

        if len(prophet_df) < 50:
            print(f"Skipping {company} due to insufficient data.")
            continue

        # Step 3: Train-Test Split
        split = int(0.8 * len(prophet_df))
        train = prophet_df[:split]
        test = prophet_df[split:]

        # Step 4: Train Prophet model
        model = Prophet(daily_seasonality=True)
        model.fit(train)

        # Step 5: Forecast on test period
        future = model.make_future_dataframe(periods=len(test))
        forecast = model.predict(future)

        y_pred = forecast.iloc[-len(test):]["yhat"].values
        y_true = test["y"].values

        mse = mean_squared_error(y_true, y_pred)
        results_prophet.append((company, "Prophet", mse))

        # Step 6: Plot results
        plt.figure(figsize=(10, 4))
        plt.plot(test["ds"], y_true, label="Actual")
        plt.plot(test["ds"], y_pred, label="Forecast", color="red")
        plt.title(f"{company} - Prophet Forecast")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Skipping {company} due to Prophet error: {e}")

# STEP 7 - LSTM Model Forecasting

from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def create_lstm_data(data, window=60):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

for company in companies[:50]:
    try:
        company_df = df[df["Company"] == company]
        close = company_df["Close"].values.reshape(-1, 1)

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(close)

        X, y = create_lstm_data(scaled)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)

        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)
        y_actual = scaler.inverse_transform(y.reshape(-1, 1))
        mse = mean_squared_error(y_actual, predictions)
        results.append((company, "LSTM", mse))

        plt.figure(figsize=(10, 4))
        plt.plot(y_actual, label="Actual")
        plt.plot(predictions, label="Predicted")
        plt.title(f"{company} - LSTM Forecast")
        plt.legend()
        plt.grid()
        plt.show()
    except:
        print(f"Skipping {company} due to LSTM error.")

# STEP 8 - Combining and Visualizing the Models

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Combine results
combined_results = results + results_sarima_statsmodels + results_prophet
results_df = pd.DataFrame(combined_results, columns=["Company", "Model", "MSE"])

# Sort for cleaner display
results_df = results_df.sort_values(by=["Company", "Model"])

# Plot
plt.figure(figsize=(16, 8))
sns.set(style="whitegrid")

# Thicker bars, custom palette
bar_plot = sns.barplot(
    data=results_df,
    x="Company",
    y="MSE",
    hue="Model",
    palette="Set2",
    width=1.0,
    dodge=True
)

# Optional: Use log scale to compress high MSE values
bar_plot.set_yscale("log")

# Titles and labels
plt.title("Model Comparison (Log MSE per Company)", fontsize=16)
plt.xlabel("Company", fontsize=12)
plt.ylabel("MSE (log scale)", fontsize=12)

# Customize x-axis
plt.xticks(rotation=45, ha='right', fontsize=8)  # smaller font
plt.yticks(fontsize=10)

# Adjust legend
plt.legend(title="Model", fontsize=10, title_fontsize=11, loc="upper right")

plt.tight_layout()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.show()

# STEP 9 - Displaying Best Model per Company

best_models = results_df.loc[results_df.groupby("Company")["MSE"].idxmin()].sort_values(by="MSE")

print("Best Model per Company:")
print(best_models)

# Count how many times each model had the lowest MSE
model_win_counts = best_models["Model"].value_counts()
print("\nModel Win Counts:")
print(model_win_counts)

# STEP 10 - Boxplot of Model Performance

plt.figure(figsize=(10, 6))
sns.boxplot(data=results_df, x="Model", y="MSE", palette="Set2")
plt.yscale("log")  # Optional: log scale for better visibility
plt.title("Model Performance Distribution (Log MSE)", fontsize=14)
plt.xlabel("Model", fontsize=12)
plt.ylabel("MSE (log scale)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# STEP 11- Exporting to CSV

results_df.to_csv("Model Comparison Results.csv", index=False)
print("Tuned model results exported to 'Model Comparison Results.csv'")

from google.colab import files
files.download("Model Comparison Results.csv")
