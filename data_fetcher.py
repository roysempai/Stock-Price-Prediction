"""
Real-time Stock Data Fetcher
Supports multiple sources for reliable data fetching
"""

import pandas as pd
import yfinance as yf
import requests
import json
from datetime import datetime, timedelta
import logging

logging.getLogger('yfinance').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)


class StockDataFetcher:
    """Fetches real-time stock data from multiple sources"""
    
    def __init__(self):
        # You can get free API keys from:
        # Alpha Vantage: https://www.alphavantage.co/api/
        # Finnhub: https://finnhub.io/
        self.alpha_vantage_key = None  # Set your key here
        self.finnhub_key = None        # Set your key here
    
    def fetch_from_yfinance(self, ticker, period='2y'):
        """Fetch data from yfinance"""
        try:
            print(f"   Trying yfinance...")
            data = yf.download(ticker, period=period, progress=False)
            if not data.empty and len(data) > 50:
                print(f"   [OK] yfinance successful!")
                return data
        except Exception as e:
            print(f"   [WARN] yfinance failed")
        return None
    
    def fetch_from_alpha_vantage(self, ticker):
        """Fetch data from Alpha Vantage (free tier: 5 calls/min, 500/day)"""
        if not self.alpha_vantage_key:
            return None
        
        try:
            print(f"   Trying Alpha Vantage...")
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': ticker,
                'apikey': self.alpha_vantage_key,
                'outputsize': 'full'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data_json = response.json()
            
            if 'Time Series (Daily)' in data_json:
                time_series = data_json['Time Series (Daily)']
                dates = []
                closes = []
                
                for date_str, values in time_series.items():
                    dates.append(pd.to_datetime(date_str))
                    closes.append(float(values['4. close']))
                
                df = pd.DataFrame({'Close': closes}, index=dates)
                df.sort_index(inplace=True)
                print(f"   [OK] Alpha Vantage successful!")
                return df
        except Exception as e:
            print(f"   [WARN] Alpha Vantage failed: {e}")
        
        return None
    
    def fetch_from_finnhub(self, ticker):
        """Fetch data from Finnhub"""
        if not self.finnhub_key:
            return None
        
        try:
            print(f"   Trying Finnhub...")
            url = f"https://finnhub.io/api/v1/quote"
            params = {'symbol': ticker, 'token': self.finnhub_key}
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'c' in data:  # Current price
                print(f"   [OK] Finnhub successful!")
                return float(data['c'])
        except Exception as e:
            print(f"   [WARN] Finnhub failed: {e}")
        
        return None
    
    def fetch_stock_data(self, ticker, period='2y'):
        """
        Try multiple sources in order of reliability
        Returns: pandas DataFrame with stock data
        """
        print(f"\n[DATA] Fetching real-time data for {ticker}...")
        
        # Try yfinance first (most reliable)
        data = self.fetch_from_yfinance(ticker, period)
        if data is not None:
            return data, "yfinance"
        
        # Try Alpha Vantage
        data = self.fetch_from_alpha_vantage(ticker)
        if data is not None:
            return data, "Alpha Vantage"
        
        # Try Finnhub (returns single price, not historical)
        price = self.fetch_from_finnhub(ticker)
        if price is not None:
            print(f"   [WARN] Only current price available: INR {price:.2f}")
            return None, "Finnhub"
        
        # All sources failed
        print(f"   [ERROR] All real-time sources failed")
        return None, "Failed"


def setup_alpha_vantage_key(api_key):
    """Configure Alpha Vantage API key"""
    fetcher = StockDataFetcher()
    fetcher.alpha_vantage_key = api_key
    return fetcher


def setup_finnhub_key(api_key):
    """Configure Finnhub API key"""
    fetcher = StockDataFetcher()
    fetcher.finnhub_key = api_key
    return fetcher
