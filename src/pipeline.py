# src/pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df, features, target):
    df = df[features + [target]].dropna()
    X = df[features]
    y = df[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R²:", r2_score(y_test, y_pred))

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted")
    plt.show()

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def main():
    # Step 1: Load data
    df = load_data("data/train.csv")

    # Step 2: Preprocess
    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']
    target = 'SalePrice'
    X, y, scaler = preprocess_data(df, features, target)

    # Step 3: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 4: Train
    model = train_model(X_train, y_train)

    # Step 5: Evaluate
    evaluate_model(model, X_test, y_test)

    # Step 6: Save model
    save_model(model, "model/house_price_model.pkl")

if __name__ == "__main__":
    main()
