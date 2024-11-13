import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Title of the app
st.title("Stock Price Predictor App 📈")

# User input for stock symbol
stock = st.text_input("Enter the Stock Symbol (e.g., GOOG, AAPL, MSFT)", "GOOG")

# Setting start and end dates for historical data
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

# Fetching stock data from Yahoo Finance
try:
    stock_data = yf.download(stock, start, end)
    st.success(f"Data successfully loaded for {stock}!")
except Exception as e:
    st.error(f"Failed to load stock data: {e}")
    st.stop()

# Load pre-trained model
try:
    model = load_model("stock_price_model.keras")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Display stock data
st.subheader(f"Stock Data for {stock}")
st.write(stock_data)

# Splitting data into training and testing
splitting_len = int(len(stock_data) * 0.7)
x_test = stock_data[['Close']].iloc[splitting_len:].copy()

# Preprocessing data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i - 100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Make predictions
predictions = model.predict(x_data)

# Inverse transform to get original values
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)



# Preparing data for plotting
plotting_data = pd.DataFrame(
    {
        'Original Test Data': inv_y_test.reshape(-1),
        'Predictions': inv_pre.reshape(-1),
    },
    index=stock_data.index[splitting_len + 100:],
)

# Plot original vs predicted values
st.subheader("Original vs Predicted Values")
st.write(plotting_data)

st.subheader("Original Close Price vs Predicted Close Price")
fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([stock_data['Close'][:splitting_len + 100], plotting_data], axis=0))
plt.legend(["Data (not used for testing)", "Original Test Data", "Predicted Test Data"])
plt.title(f"Close Price vs Predictions for {stock}")
st.pyplot(fig)

# --- PREDICTING TOMORROW'S STOCK PRICE ---
# Get the last 100 days' closing prices for the prediction
last_100_days = stock_data['Close'].iloc[-100:].values

# Reshape and scale the data for the prediction
last_100_days_scaled = scaler.transform(last_100_days.reshape(-1, 1))

# Prepare the input data in the same format as used for the model
X_test_today = []
X_test_today.append(last_100_days_scaled)
X_test_today = np.array(X_test_today)

# Predict the next day's stock price
predicted_tomorrow = model.predict(X_test_today)
predicted_tomorrow_price = scaler.inverse_transform(predicted_tomorrow)

# Display the predicted value for tomorrow in the sidebar
st.sidebar.subheader(f"Prediction for Tomorrow's ({stock}) Closing Price")
st.sidebar.write(f"The predicted closing price for **{stock}** tomorrow is: **${predicted_tomorrow_price[0][0]:.2f}**")


# Calculate MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((inv_y_test - inv_pre) / inv_y_test)) * 100
accuracy = 100 - mape

# Display accuracy
st.sidebar.subheader("Prediction Accuracy")
st.sidebar.write(f"The model's prediction accuracy is approximately **{accuracy:.2f}%**.")