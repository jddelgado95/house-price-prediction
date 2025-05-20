
#matplotlib, seaborn: for data visualization.
import matplotlib.pyplot as plt
import seaborn as sns
#sklearn: for machine learning:
#train_test_split: split data into train/test sets.
from sklearn.model_selection import train_test_split
#LinearRegression: the prediction model.
from sklearn.linear_model import LinearRegression
#mean_squared_error, r2_score: performance metrics.
from sklearn.metrics import mean_squared_error, r2_score
#os: for handling file paths and directories.
import os
##joblib: to save the trained model.
import joblib
#import functions from utils
from utils import load_data, preprocess_data, save_model

#Initializes and trains a Linear Regression model on the training data.
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

#Evaluate the Model
def evaluate_model(model, X_test, y_test):
    #Uses the trained model to predict on test data.
    y_pred = model.predict(X_test)
    #MSE (mean squared error): how far off predictions are.
    #R² score: how well the model explains variance in data (1 = perfect).
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R²:", r2_score(y_test, y_pred))
    #plots predicted vs actual values.
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted")
    plt.show()

#Creates the directory (if it doesn’t exist).
#Saves the trained model to a file using joblib
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def main():
    # Step 1: Load data
    df = load_data("data/train.csv")

    # Step 2: Preprocess: Prepares the data.
    features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']
    target = 'SalePrice'
    X, y, scaler = preprocess_data(df, features, target)

    # Step 3: Split: Splits it into training/testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 4: Train: Trains the model.
    model = train_model(X_train, y_train)

    # Step 5: Evaluates it.
    evaluate_model(model, X_test, y_test)

    # Step 6: Saves the trained model to disk.
    save_model(model, "model/house_price_model.pkl")
    save_model(scaler, "model/scaler.pkl")

#This ensures the script runs the pipeline only when executed directly, not when imported as a module.
if __name__ == "__main__":
    main()
