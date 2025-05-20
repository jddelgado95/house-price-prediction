#pandas, numpy: for data manipulation.
import pandas as pd
##joblib: to save the trained model.
import joblib
##StandardScaler: feature normalization.
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load a CSV dataset from disk."""
    return pd.read_csv(filepath)

def preprocess_data(df, features, target):
    """
    Selects features and target, drops missing values, and scales features.
    
    Returns scaled features, target, and the fitted scaler.
    """
    df = df[features + [target]].dropna()
    X = df[features]
    y = df[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def save_model(obj, path):
    """Saves a model or object to disk using joblib."""
    joblib.dump(obj, path)

def load_model(path):
    """Loads a model or object from disk using joblib."""
    return joblib.load(path)


##Reads a CSV file from the data/ folder into a pandas DataFrame
#def load_data(filepath):
    #df = pd.read_csv(filepath)
    #return df

#Selects features + target from the DataFrame.
#def preprocess_data(df, features, target):
    #Drops missing rows (to simplify the model).
    #df = df[features + [target]].dropna()
    #X: feature matrix; y: target (house prices).
    #X = df[features]
    #y = df[target]
    
    #scaler = StandardScaler()
    #Standardizes the features using StandardScaler (mean = 0, std = 1).
    #X_scaled = scaler.fit_transform(X)
    #Returns:
        #X_scaled: the normalized features
        #y: target values
        #scaler: in case you want to reuse it later (e.g. in production)
    #return X_scaled, y, scaler