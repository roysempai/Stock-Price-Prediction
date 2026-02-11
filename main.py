#!/usr/bin/env python3
"""
Main entry point for HINDCOPPER stock prediction (hackathon-ready).
Run: python main.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import multi-source data fetcher
from data_fetcher import StockDataFetcher

def generate_synthetic_data(ticker, periods=500):
    dates = pd.date_range(end=pd.Timestamp.now(), periods=periods)
    np.random.seed(hash(ticker) % 2**32)
    price_map = {
        'HINDCOPPER': 578,
        'TATAPOWER': 365.4,
        'GODFRYPHLP': 850,
        'SILVERBEES': 700,
    }
    target_price = price_map.get(ticker.upper(), 450)
    returns = np.random.normal(0.0003, 0.015, periods)
    prices = target_price * np.exp(np.cumsum(returns))
    prices *= (target_price / prices[-1])
    return pd.DataFrame({
        'Close': prices,
        'Volume': np.random.randint(1_000_000, 10_000_000, periods),
        'Open': prices * (1 + np.random.normal(0, 0.01, periods)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.02, periods))),
        'Low':  prices * (1 - np.abs(np.random.normal(0, 0.02, periods))),
    }, index=dates)

def fetch_data(ticker):
    fetcher = StockDataFetcher()
    try:
        data, source = fetcher.fetch_stock_data(ticker, period='2y')
        if data is not None and len(data) >= 100:
            print(f"✅ Real data from {source}! ({len(data)} trading days)")
            return data, source
        else:
            print("⚠️ Insufficient real data, using synthetic fallback.")
    except Exception as e:
        print(f"⚠️ Fetch failed ({e}). Using synthetic fallback.")
    return generate_synthetic_data(ticker, periods=500), "synthetic"

def train_and_predict(data, days_ahead=30):
    df = data.copy()
    df['Day'] = np.arange(len(df))
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df.dropna(inplace=True)

    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    X_train = train[['Day', 'SMA_5', 'SMA_20']].values
    y_train = train['Close'].values
    X_test  = test[['Day', 'SMA_5', 'SMA_20']].values
    y_test  = test['Close'].values

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # 30-day forecast (simple continuation using last SMAs)
    last_day = df['Day'].iloc[-1]
    last_sma5 = df['SMA_5'].iloc[-1]
    last_sma20 = df['SMA_20'].iloc[-1]

    future_days = np.arange(last_day + 1, last_day + 1 + days_ahead)
    X_future = np.column_stack([future_days,
                                np.full(days_ahead, last_sma5),
                                np.full(days_ahead, last_sma20)])
    future_preds = model.predict(X_future)

    return preds, future_preds, rmse, r2, df, test

def plot_results(df, test, preds, future_preds, ticker, source):
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], label='Actual Price')
    plt.plot(test.index, preds, label='Predicted Price')
    future_index = pd.date_range(start=df.index[-1], periods=len(future_preds)+1, freq='D')[1:]
    plt.plot(future_index, future_preds, linestyle='--', label='30-Day Forecast')
    plt.title(f"{ticker.upper()} (NSE) - Predictions vs Actual | Source: {source.upper()}")
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.tight_layout()
    out = f"{ticker.upper()}_demo.png"
    plt.savefig(out, dpi=160)
    plt.show()
    print(f"📸 Saved plot: {out}")

def main():
    ticker = input("Enter NSE ticker (e.g., HINDCOPPER): ").strip().upper()
    data, source = fetch_data(ticker)
    preds, future_preds, rmse, r2, df, test = train_and_predict(data, days_ahead=30)
    print(f"\n📊 Metrics → RMSE: {rmse:.2f}, R²: {r2:.3f}")
    print(f"🔮 30-Day Forecast (last value): ₹{future_preds[-1]:.2f}")
    plot_results(df, test, preds, future_preds, ticker, source)

if __name__ == '__main__':
    main()
