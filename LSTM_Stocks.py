import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf

# Load stock data (using Yahoo Finance)
yf.pdr_override()
def load_data(ticker, start, end):
    return pdr.get_data_yahoo(ticker, start=start, end=end)

# Preprocess data
def preprocess_data(data):
    # Use closing prices
    data = data[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Create a dataset for LSTM (X_train, y_train)
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Build LSTM Model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Plot predictions vs actual values
def plot_predictions(y_true, y_pred):
    plt.plot(y_true, label='Actual Price')
    plt.plot(y_pred, label='Predicted Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Load historical stock data
ticker = 'AAPL'
start_date = '2015-01-01'
end_date = '2020-01-01'
data = load_data(ticker, start=start_date, end=end_date)

# Preprocess the data
scaled_data, scaler = preprocess_data(data)

# Create training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Prepare data for LSTM model
X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

# Reshape data for LSTM input [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build and train the LSTM model
lstm_model = build_lstm_model((X_train.shape[1], 1))
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32)

# Predict stock prices
predicted_stock_price = lstm_model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price.reshape(-1, 1))

# Actual stock prices
actual_stock_price = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot predictions vs actual
plot_predictions(actual_stock_price, predicted_stock_price)
