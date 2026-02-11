# Stock Trend Forecasting using Logistic Regression

## Project Overview

This project focuses on predicting the next-day stock price movement (UP or DOWN) using Machine Learning.

We are using:
- Daily stock data from Yahoo Finance
- Technical indicators (EMA, RSI, Returns, Volatility)
- Logistic Regression for classification

The goal is to evaluate how well a classical ML model can classify stock trends.

---

## Objective

To build a binary classification model that predicts:

- 1 → Stock price will go UP tomorrow  
- 0 → Stock price will go DOWN tomorrow  

We are using historical daily data of:

- RELIANCE.NS (NSE Listed Company)

---

## Why This Project?

Stock markets are noisy and difficult to predict.  
Instead of predicting the exact price (regression), we predict the direction, which is:

- More practical  
- More realistic  
- More aligned with current research trends  

This project demonstrates:
- Time-series handling  
- Feature engineering  
- Classification modeling  
- Model evaluation using proper metrics  

---

## Project Structure

Stock_Trend_Project/
│
├── data/ # Downloaded stock data (not pushed to GitHub)
├── src/
│ └── data_download.py # Script to download stock data
├── models/ # Saved models (future use)
├── reports/ # Project reports and documentation
├── requirements.txt # Required Python libraries
└── README.md

---

## Setup Instructions

### 1. Clone the Repository

git clone https://github.com/Kuu587672/clg-project-data-science.git
cd clg-project-data-science

---

### 2. Create Virtual Environment

python -m venv venv

Activate:

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

---

### 3. Install Required Libraries

pip install -r requirements.txt

---

## Download Stock Data

We do not store raw stock data in GitHub.

To download the dataset:
python src/data_download.py

This will:

- Download daily stock data for RELIANCE.NS  
- Save it inside the `data/` folder  

Data source: Yahoo Finance

---

## Workflow Summary

The project follows these steps:

1. Download historical daily stock data  
2. Clean the dataset  
3. Create technical indicators:
   - EMA (Exponential Moving Average)  
   - RSI (Relative Strength Index)  
   - Daily Returns  
   - Volatility  
4. Create Target variable:
   - 1 → Next day price increases  
   - 0 → Next day price decreases  
5. Split dataset using time-based split  
6. Train Logistic Regression model  
7. Evaluate using:
   - Accuracy  
   - Confusion Matrix  
   - ROC Curve  
   - Precision & Recall  

---

## Why Logistic Regression?

Logistic Regression is used because:

- It is a strong baseline classifier  
- Easy to interpret  
- Widely used in academic research  
- Works well for binary classification  

This model helps us understand which indicators influence stock movement direction.

---

## Important Notes

- This project uses daily data (not minute-level) to reduce noise.  
- Time-series split is used instead of random shuffle.  
- Accuracy above 60% is considered strong in stock prediction problems.

---

## Team Project

This is a collaborative group project.  
All team members should:

1. Clone the repository  
2. Install dependencies  
3. Run the data download script  
4. Continue development in feature branches  

---

## Future Improvements

- Compare with Random Forest  
- Compare with SVM  
- Add XGBoost  
- Perform hyperparameter tuning  
- Add cross-validation for time-series  

---

## Disclaimer

This project is for academic and research purposes only.  
It is not financial advice.
