# 📈 Time Series Analysis and Forecasting for Stock Market

This project explores time series forecasting of stock prices using Yahoo Finance data, utilizing both classical and deep learning models. The project implements and compares four different forecasting approaches:

- **ARIMA** (AutoRegressive Integrated Moving Average)
- **SARIMA** (Seasonal ARIMA)
- **Facebook Prophet**
- **LSTM** (Long Short-Term Memory)

Model performance is evaluated and compared using Mean Squared Error (MSE) across multiple companies for robust benchmarking.

## 📂 Dataset

- **Source**: Yahoo Finance
- **File**: `Yahoo Finance Stock Data.csv`
- **Key Columns**:
  - `Date`: Time series index
  - `Company`: Stock ticker/company identifier
  - `Close`: Target column for prediction (closing price)

## 🛠️ Installation

Make sure to install all necessary dependencies before running the notebook:

```bash
pip install pmdarima prophet keras yfinance pandas scikit-learn matplotlib seaborn
```

## 🔄 Workflow

### 1. 📥 Dataset Upload & Cleaning

- Dataset is uploaded using Google Colab's `files.upload()` method
- Columns are cleaned by stripping whitespace and standardizing formats
- The `Date` column is parsed using `pd.to_datetime()` with UTC handling
- Data is sorted by `Company` and `Date` to ensure correct temporal order
- The cleaned dataset is saved as `Cleaned Yahoo Finance Stock Data.csv`

### 2. 📊 Data Visualization

- Line plots are created to visualize closing prices over time
- Each company's closing price is plotted individually to identify trends, seasonality, and anomalies

### 3. 🔁 ARIMA Forecasting

- ARIMA is a classical time series model implemented using `statsmodels`
- **Model Configuration**: ARIMA(5,1,0)
- **Data Split**: 
  - 80% for training
  - 20% for testing
- ARIMA model is fit to the training set and forecasted on the test set
- MSE is calculated to evaluate performance
- Forecasts are visualized along with actual prices

### 4. 🔁 SARIMA Forecasting

- SARIMA (Seasonal ARIMA) extends ARIMA to capture seasonality
- **Configuration**: SARIMA(1,1,1)(1,1,1,12)
- Uses `SARIMAX` from `statsmodels`
- Handles monthly seasonality (s=12)
- Same data split as ARIMA (80/20)
- MSE calculation and forecast vs actual plots are generated

### 5. 🔮 Prophet Forecasting

- Facebook Prophet is designed to handle time series with strong seasonal effects and missing data
- **Data Formatting**: 
  - `ds` for date
  - `y` for value (closing price)
- Model trained on 80% of data, predictions made for remaining 20%
- Prophet automatically handles:
  - Trends
  - Seasonality (weekly, yearly)
  - Holiday effects
- MSE is computed and visualizations are generated

### 6. 🤖 LSTM Neural Network Forecasting

LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) specifically designed for modeling sequential data like time series.

#### 🔧 Preprocessing
- Close prices are scaled using `MinMaxScaler`
- A sliding window approach creates sequences of `n_steps` (e.g., 30)
- Data is reshaped to fit LSTM input format: `(samples, time steps, features)`

```python
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
```

#### 🏗️ Model Architecture

```python
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(n_steps, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
```

#### ⚙️ Training
- Model is trained on the training split of the time series
- Batch size, epochs, and learning rate can be customized
- Predictions are inverse transformed back to original scale

#### 📏 Evaluation
- Predictions are compared with actual test values
- MSE is used as the primary evaluation metric
- Forecasts are plotted for visual inspection

### 7. 📊 Results Compilation

- MSE results for all models (ARIMA, SARIMA, Prophet, LSTM) are stored in a DataFrame
- **Columns**: Company, Model, MSE
- Results are combined and sorted for comparative analysis
- Final output is exported as `Model Comparison Results.csv`

## ✅ Output Files

- `Cleaned Yahoo Finance Stock Data.csv` – Cleaned and processed dataset
- `Model Comparison Results.csv` – Combined MSE results for all models and companies
- Time series plots – Forecast vs Actual visualizations for each company and model

## 🧠 Future Enhancements

- 🔁 Implement rolling forecast windows (sliding test sets)
- ⚡ Hyperparameter tuning using `optuna` or `KerasTuner`
- 📊 Integrate technical indicators (e.g., RSI, MACD, Bollinger Bands)
- 🧠 Experiment with Transformer or attention-based models for time series
- 🏦 Add support for additional data sources (e.g., Alpha Vantage, Quandl)

## 🚀 Getting Started

1. Clone this repository
2. Install the required dependencies
3. Upload your Yahoo Finance stock data CSV file
4. Run the notebook cells sequentially
5. Review the generated comparison results and visualizations

---

**Note**: This project is for educational and research purposes. Please do not use the predictions for actual trading decisions without proper risk assessment and additional validation.