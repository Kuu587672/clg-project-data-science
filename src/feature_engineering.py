import pandas as pd
import numpy as np

def feature_engineering():
    
    # Load clean dataset
    data = pd.read_csv("data/reliance_daily.csv", parse_dates=['Date'])
    data.set_index('Date', inplace=True)
    
    # Daily return
    data["Return"] = data["Close"].pct_change()
    
    # EMA (Exponential Moving Average)
    data["EMA_10"] = data["Close"].ewm(span=10).mean()
    data["EMA_20"] = data["Close"].ewm(span=20).mean()
    
    # Volatility
    data["Volatility"] = data["High"] - data["Low"]
    
    # RSI (Relative Strength Index)
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    
    # Target variable
    data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
    
    # Dropping missing values caused by indicators
    data.dropna(inplace=True)
    
    # Save processed dataset
    data.to_csv("data/reliance_processed.csv")
    
    print("Feature engineering completed.")
    print("\nFinal dataset shape: ", data.shape)
    print("\nColumns: ")
    print(data.columns)
    
    return data

if __name__ == "__main__":
    feature_engineering()