# House Price Prediction

A simple regression model to predict house prices based on features like size, quality, and garage capacity.

## Tech Stack

- Python
- pandas, NumPy, scikit-learn
- matplotlib, seaborn
- flask

## how to run

$ Run a virtual environment:

Run requirements:
$ pip install -r requirements.txt

Train the model:
$ python3 src/pipeline.py

You should see these two files with the model and the scaler:
-rw-r--r-- 1 house_price_model.pkl
-rw-r--r-- 1 scaler.pkl

Run the Flask web app:
PYTHONPATH=. python3 app/app.py

Running on http://127.0.0.1:5000

If you want to avoid using the PYTHONPATH variable, you could make the project a package by adding an **init**.py file:

$touch src/**init**.py

To get the train.csv data, you should:

1. Log in to Kaggle (create a free account if you don't have one).
2. Go to the dataset link above.
3. Click the "Download All" button.
4. Unzip the file and locate train.csv.
5. Move train.csv into the project’s data/ folder:
   house-price-prediction/
   ├── data/
   │ └── train.csv //put it here
