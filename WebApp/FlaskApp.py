import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

import pickle
import os

app = Flask(__name__)
model_path = os.path.realpath("serialized_models/LogisticRegression")
model = pickle.load(open(model_path + "/logistic_regression.mdl", mode='rb'))
imputer = pickle.load(open(model_path + "/imputer.imp", mode='rb'))

@app.route('/')
def home():
    return render_template('index.html')

#Note: The value of route and the name of the function must be the same
@app.route('/predictHTML',methods=['POST'])
def predictHTML():
    
    for i in request.form.values():
        print(i)
    int_features = [x for x in request.form.values()]
    ##final_features = [np.array(int_features)]
    ##prediction = model.predict(final_features)
    return render_template('index.html', prediction_text=str(int_features))
    ##output = round(prediction[0], 2)

    #return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        threshold = .44 #The threshold used to classify a result (if p[x=="Positive"] > .44, then x is "Positive")
        
        dataJson = request.get_json(force=True)
        dataFormatted = pd.read_json(dataJson) #If we are going to predict many. I need to create another endpoint if we are just going to predict 1
        imputer.transform(dataFormatted)
        prediction = model.predict_proba(dataFormatted) #Need also to implement the prediction threshold here 

        # pd.cut() is used to bin the probabilities
        # if the probability (of class positive) is between 0-threshold then, the label assigned is "Negative"
        # 'right=False' means to classify the threshold as a "Negative" result (ie if p[x] == threshold, then x is "Negative")
        prediction = pd.cut(prediction[:,1], [0, threshold, 1], labels=["Negative", "Positive"], right=False) 
        print(prediction)
        return jsonify(prediction.tolist())
    except Exception as e:
        message = "An error ocurred while making a prediction: " + str(e)
        return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)