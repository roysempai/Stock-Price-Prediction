# 📈 HINDCOPPER Stock Price Prediction (Hackathon Project)

Predicts HINDCOPPER (NSE) stock prices using Machine Learning with real-time data + safe fallback for demos.

## 🚀 Features
- Real-time data (Alpha Vantage / Finnhub / yfinance)
- Automatic fallback to realistic synthetic data (demo never fails)
- ML prediction + 30-day forecast
- Clean plots for judges

## 🛠️ Tech
Python, Pandas, NumPy, scikit-learn, Matplotlib, yfinance

## 📦 Install
```bash
pip install -r requirements.txt
```

## ▶️ Run (Judges)
```bash
python main.py
```
Enter:
```
HINDCOPPER
```

## 🔑 Optional: Real-Time API (2 minutes)
1) Get free key: https://www.alphavantage.co/api/  
2) Open `data_fetcher.py` and set:
```python
self.alpha_vantage_key = "YOUR_API_KEY"
```
If APIs fail, the app uses realistic fallback automatically.

## 📊 Output
- Actual vs Predicted price curve
- 30-day forecast (dashed line)
- Plot auto-saved as `HINDCOPPER_demo.png`

## ⚠️ Disclaimer
Educational/hackathon use only. Not financial advice.
