# Full Exploratory Data Analysis on reliance_daily.csv
# Following structured EDA steps (as per standard EDA workflow)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1. Data Collection & Loading
# -----------------------------
data = pd.read_csv("../data/reliance_daily.csv")

print("===== FIRST 5 ROWS =====")
print(data.head())

print("\n===== DATA INFO =====")
print(data.info())

print("\n===== DESCRIPTIVE STATISTICS =====")
print(data.describe())

# -----------------------------
# 2. Missing Values & Duplicates
# -----------------------------
print("\n===== MISSING VALUES =====")
print(data.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(data.duplicated().sum())

# -----------------------------
# 3. Data Formatting
# -----------------------------
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)
data.set_index('Date', inplace=True)

# -----------------------------
# 4. Feature Engineering
# -----------------------------
data['Return'] = data['Close'].pct_change()
data['Volatility'] = data['Return'].rolling(10).std()
data['EMA10'] = data['Close'].ewm(span=10).mean()
data['EMA20'] = data['Close'].ewm(span=20).mean()

# -----------------------------
# 5. Univariate Analysis
# -----------------------------
plt.figure()
plt.hist(data['Close'], bins=50)
plt.title("Close Price Distribution")
plt.xlabel("Close Price")
plt.ylabel("Frequency")
plt.show()

plt.figure()
plt.hist(data['Volume'], bins=50)
plt.title("Volume Distribution")
plt.xlabel("Volume")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# 6. Time Series Analysis
# -----------------------------
plt.figure()
plt.plot(data['Close'])
plt.title("Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()

# -----------------------------
# 7. Outlier Detection (Return)
# -----------------------------
plt.figure()
plt.boxplot(data['Return'].dropna())
plt.title("Return Outliers")
plt.ylabel("Return")
plt.show()

# -----------------------------
# 8. Correlation Analysis
# -----------------------------
corr = data.corr()

plt.figure()
plt.imshow(corr)
plt.title("Correlation Matrix")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar()
plt.show()

print("\n===== CORRELATION MATRIX VALUES =====")
print(corr)

# -----------------------------
# 9. Aggregation Example (Yearly Avg Close)
# -----------------------------
yearly_avg = data['Close'].resample('Y').mean()

plt.figure()
plt.plot(yearly_avg)
plt.title("Yearly Average Closing Price")
plt.xlabel("Year")
plt.ylabel("Average Close Price")
plt.show()

print("\n===== EDA COMPLETE =====")
