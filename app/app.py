from flask import Flask, render_template, request
from src.utils import load_model
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = load_model("model/house_price_model.pkl")
scaler = load_model("model/scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract user inputs from the form
    try:
        input_data = [
            int(request.form['OverallQual']),
            int(request.form['GrLivArea']),
            int(request.form['GarageCars']),
            int(request.form['TotalBsmtSF'])
        ]

        # Prepare and scale input
        features = np.array([input_data])
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction=prediction)

    except ValueError:
        return render_template('index.html', prediction="Invalid input.")

if __name__ == '__main__':
    app.run(debug=True)