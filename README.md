# 📈 Time Series Analysis and Forecasting for Stock Market

This project explores **time series forecasting** of stock prices using Yahoo Finance data, utilizing both classical and deep learning models - 
- **ARIMA**
- **SARIMA**
- **Facebook Prophet**
- **LSTM (Long Short-Term Memory)**

Model performance is compared using **Mean Squared Error (MSE)**. All models are applied across multiple companies for robust evaluation and performance benchmarking.

---

## 📂 Dataset

- **Source** - Yahoo Finance  
- **File Used** - `Yahoo Finance Stock Data.csv`  
- **Key Columns** -
  - `Date`
  - `Company`
  - `Close` (target column for prediction)

---

## 🛠️ Workflow Steps

### 1. 📦 Install Dependencies

Make sure to install all necessary libraries before running the notebook:

```bash
pip install pmdarima prophet keras yfinance pandas scikit-learn matplotlib seaborn

---

### 2. 📥 Dataset Upload & Cleaning

- Dataset is uploaded using Google Colab's files.upload() method.

- Columns are cleaned by stripping whitespace and standardizing formats.

- The Date column is parsed using pd.to_datetime() with UTC handling.

- Data is sorted by Company and Date to ensure correct temporal order.

- The cleaned dataset is saved as - Cleaned Yahoo Finance Stock Data.csv.

---

### 3. 📊 Data Visualization

- Line plots are created to visualize the closing prices over time.

- Each company's closing price is plotted individually to identify trends, seasonality or anomalies.

---

### 4. 🔁 ARIMA Forecasting

- ARIMA is a classical time series model implemented using statsmodels.

- Model configuration - ARIMA(5,1,0)

- Each company’s closing price series is split -

  - 80% for training

  - 20% for testing

- ARIMA model is fit to the training set and forecasted on the test set.

- MSE (Mean Squared Error) is calculated to evaluate performance.

- Forecasts are visualized along with actual prices.

---

### 5. 🔁 SARIMA Forecasting

- SARIMA (Seasonal ARIMA) extends ARIMA to capture seasonality.

- Configuration used - SARIMA(1,1,1)(1,1,1,12)

- Uses SARIMAX from statsmodels.

- Data is split the same way as ARIMA.

- Handles monthly seasonality (s=12).

- MSE is calculated and forecast vs actual plots are generated.

---

### 6. 🔮 Prophet Forecasting

- Facebook Prophet is designed to handle time series with strong seasonal effects and missing data.

- Data is formatted to Prophet’s required format -

  - ds for date

  - y for value (closing price)

- Model is trained on 80% of data, predictions are made for the remaining 20%.

- Prophet automatically handles -

  - Trends

  - Seasonality (weekly, yearly)

  - Holiday effects

- MSE is computed and visualizations are generated.

---

### 7. 🤖 LSTM Neural Network Forecasting

- LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) used for modeling time series data.

- 🔧 Preprocessing

  - Close prices are scaled using MinMaxScaler.

  - A sliding window approach is used to create sequences of n_steps (e.g., 30).

  - Data is reshaped to fit LSTM input - (samples, time steps, features).

```bash
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

- 🏗 Model Architecture

```bash
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(n_steps, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


- ⚙️ Training

  - Model is trained on the training split of the time series.

  - Batch size, epochs, and learning rate can be customized.

  - Predictions are inverse transformed back to original scale.

- 📏 Evaluation
  - Predictions are compared with actual test values.

  - MSE is used as the primary evaluation metric.

  - Forecasts are plotted for visual inspection.

---

### 8. 📊 Results Compilation

- MSE results for all models (ARIMA, SARIMA, Prophet, LSTM) are stored in a DataFrame -

  - Columns - Company, Model, MSE

- Results are combined and sorted for comparative analysis.

- Final output is exported as Model Comparison Results.csv

---

## ✅ Output Files

- Cleaned Yahoo Finance Stock Data.csv – Cleaned and processed dataset.

- Model Comparison Results.csv – Combined MSE results for all models and companies.

- Time series plots – Forecast vs Actual for each company and model.

---

### 🧠 Future Enhancements

- 🔁 Implement rolling forecast windows (sliding test sets)

- ⚡ Hyperparameter tuning using optuna or KerasTuner

- 📊 Integrate technical indicators (e.g., RSI, MACD, Bollinger Bands)

- 🧠 Experiment with Transformer or attention-based models for time series

- 🏦 Add support for additional data sources (e.g., Alpha Vantage, Quandl)


