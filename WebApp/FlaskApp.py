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
    ### Test input (which results in positiv if we do not use the imputer and negative if we do)
    ### {'wheezes': 'True', 'ctab': 'False', 'high_risk_exposure_occupation': 'True', 'cough': 'False', 'loss_of_smell': 'True', 'loss_of_taste': 'False', 'muscle_sore': 'True', 'headache': 'False', 'fatigue': 'True', 'diabetes': 'False', 'pam': '1', 'rr': '2', ###'pulse': '3', 'temperature': '4', 'days_since_symptom_onset': '5'}
    
    categorical_cols = ['wheezes', 'ctab', 'high_risk_exposure_occupation', 'cough',
       'loss_of_smell', 'loss_of_taste', 'muscle_sore', 'headache', 'fatigue',
       'diabetes']
    numerical_cols = ['pam', 'rr', 'pulse', 'temperature',
       'days_since_symptom_onset']
    ordered_features = ["wheezes", "ctab", "pam", "rr", "pulse", "temperature", "high_risk_exposure_occupation", "cough", "loss_of_smell", "muscle_sore", "loss_of_taste", "headache", "days_since_symptom_onset", "fatigue", "fever", "diabetes"]
    data = pd.DataFrame(dict(request.form), index=[0]) #Because we are just prediction one result, the index is 0
    data[categorical_cols] = data[categorical_cols].astype("bool")
    data[numerical_cols] = data[numerical_cols].astype("float")    
    data["fever"] = data["temperature"] >= 38
    
    print(data.iloc[0])
    imputer.transform(data) #Note: If I use the imputer, the result becomes negative. However if I dont, the result is positive with the same values???
    
    print(data.iloc[0])
    data = data[ordered_features] #Because the order of the features matters in sklearn (does not remeber the names of the columns) and this statement will get the dataframe in the same order as 'ordered_features'
    
    
    prediction = model.predict(data)
    #return render_template('index.html', prediction_text=str(int_features))
    ##output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Your result is likely {}'.format(prediction[0]))

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