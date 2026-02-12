import pandas as pd

def load_data():
    data = pd.read_csv("data/reliance_daily.csv")
    
    # convert 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])
    
    # set date as index
    data.set_index('Date', inplace=True)
    
    print("First 5 rows of the dataset:")
    print(data.head())
    
    print("\nDataset information:")
    print(data.info())
    
    print("\nMissing values:")
    print(data.isnull().sum())
    
    return data

if __name__ == "__main__":
    df = load_data()