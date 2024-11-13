# Stock Price Prediction App 📈

Welcome to the **Stock Price Prediction App**! This web application uses machine learning and historical stock price data to predict future stock prices. The app fetches the stock data from **Yahoo Finance**, processes it using a pre-trained **LSTM model** (Long Short-Term Memory), and provides predictions for both past and future stock prices.

You can also check out the app live by clicking on the link below:

**[Live Demo](https://stock-price-prediction-murli.streamlit.app/)**

---

## Features

- **Stock Data Visualization**: View historical stock data along with moving averages for 100, 200, and 250 days.
- **Stock Price Predictions**: Get predictions for future stock prices based on historical data.
- **Tomorrow's Prediction**: The app predicts the next day's stock price and displays it in the sidebar.
- **Prediction Accuracy**: View the accuracy of the predictions, based on historical stock prices.
- **Interactive UI**: Enter any stock symbol (e.g., GOOG, AAPL, MSFT) and get real-time predictions and visualizations.

---

## How It Works

1. **Stock Symbol Input**: 
   - Enter the stock symbol (e.g., `GOOG` for Google, `AAPL` for Apple) into the input field to get predictions.
   
2. **Fetch Stock Data**:
   - The app retrieves the last 20 years of historical stock data using the **Yahoo Finance** API.

3. **Data Visualization**:
   - The app displays the stock's historical closing prices along with three moving averages (for 100, 200, and 250 days).

4. **Prediction**:
   - A pre-trained **LSTM model** (Long Short-Term Memory) is used to predict future stock prices. The model is trained on historical stock data and provides predictions.

5. **Tomorrow's Prediction**:
   - The app also predicts the next day's stock price, which is displayed in the sidebar.

6. **Prediction Accuracy**:
   - The app compares actual stock prices vs. predicted prices and calculates the prediction accuracy percentage, helping you evaluate the model's performance.

---


## Running the App Locally
1. Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
```
2. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run web_stock_price_predictor.py
```

4. Open your browser and visit the following URL: 
```arduino
http://localhost:8501
```

Now you should be able to interact with the app locally.
Feel free to explore the app and see how the model works with different stock symbols.

## Future Enhancements

Here are some potential future enhancements for the app:

### Real-Time Data: 
Integrate with real-time stock price feeds for live predictions.
### Multiple Models: 
Compare predictions from different models (e.g., Random Forest, XGBoost).
### Sentiment Analysis: 
Incorporate sentiment analysis from financial news sources to improve prediction accuracy.
### User Account: 
Allow users to save their predictions or track specific stock symbols

