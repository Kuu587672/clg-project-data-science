import yfinance as yf

def download_stock():
    data = yf.download(
        "RELIANCE.NS",
        start="2015-01-01",
        end="2025-01-01",
        interval="1d"
    )
    
    data.to_csv("data/reliance_daily.csv")
    print("Data downloaded successfully!")
    
if __name__ == "__main__":
    download_stock()