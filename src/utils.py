import joblib
import numpy as np

model = joblib.load("model/house_price_model.pkl")
scaler = joblib.load("model/scaler.pkl")

def predict_price(features_dict):
    features = np.array([[features_dict['OverallQual'],
                          features_dict['GrLivArea'],
                          features_dict['GarageCars'],
                          features_dict['TotalBsmtSF']]])
    
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return prediction[0]