import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    
    # Load processed dataset
    data = pd.read_csv("data/reliance_processed.csv", parse_dates=['Date'])
    data.set_index('Date', inplace=True)
    
    # Select features
    features = ["Return", "Return_lag1", "Return_lag2", "EMA_ratio", "RSI", "Volatility", "Volume"]
    X = data[features]
    y = data["Target"]
    
    # Time Series Cross Validation
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1000
        ))
    ])

    tscv = TimeSeriesSplit(n_splits=5)

    cv_scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=tscv,
        scoring="roc_auc"
    )

    print("\nTimeSeries Cross-Validation AUC scores:", cv_scores)
    print("Mean AUC:", np.mean(cv_scores))
    print("Std Dev:", np.std(cv_scores))
    
    # Time-based split (80% train, 20% test)
    split = int(len(data) * 0.8)
    
    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]
    
    # Scale features
    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.transform(X_test)
    
    # Train logistic regression model
    model = LogisticRegression(
        #C=0.1,                    # Regularization strength (Reduces overfitting)
        class_weight="balanced",    # Handle class imbalance
        max_iter=1000
    )
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Accuracy
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label="AUC = %0.2f" % roc_auc)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
    
    # Feature importance (coefficients)
    importance = pd.Series(model.coef_[0], index=features)
    
    plt.figure()
    importance.sort_values().plot(kind="barh")
    plt.title("Feature Importance (Logistic Regression)")
    plt.show()
    
    
if __name__ == "__main__":
    train_model()