import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

import pickle
import os

app = Flask(__name__)
model_path = os.path.realpath("serialized_models/LogisticRegression")
model = pickle.load(open(model_path + "/logistic_regression.mdl", mode='rb'))
imputer = pickle.load(open(model_path + "/imputer.imp", mode='rb'))

@app.route('/predict', methods=['POST'])
def predict():
    dataJson = request.get_json(force=True)
    dataFormatted = pd.read_json(dataJson) #If we are going to predict many. I need to create another endpoint if we are just going to predict 1
    imputer.transform(dataFormatted)
    prediction = model.predict(dataFormatted) #Need also to implement the prediction threshold here 
    print(prediction)
    return jsonify(prediction.tolist())

if __name__ == "__main__":
    app.run(debug=True)