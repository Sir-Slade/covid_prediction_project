from CovidClinicalData import read_data, clean_data, DataImputer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle
import os

##We read and clean all the data inside covidclinicaldata
training_data = read_data()
clean_data(training_data)

print("DATA LOADING COMPLETE!!!")

#Feature engineering
print("Training model....",  end=" ")

selected_features = ['cough', 'cough_severity', 'days_since_symptom_onset', 'dia',
       'diabetes', 'diarrhea', 'fatigue', 'fever', 'headache',
       'high_risk_exposure_occupation', 'loss_of_smell', 'loss_of_taste',
       'muscle_sore', 'pulse', 'rr', 'runny_nose', 'sob', 'sob_severity',
       'sore_throat', 'sys', 'temperature', 'wheezes']

#Imputation
imputer = DataImputer()
X = training_data[selected_features]
y = training_data.loc[:,"covid19_test_results"]
imputer.fit_transform(X, y)

#Resampling
smote = SMOTE()
X_oversample, y_oversample = smote.fit_resample(X, y)

#Training
model = LogisticRegression(C=2)
model.fit(X_oversample, y_oversample)

print("Complete!!!")

print("Saving model...", end="")

#Dump the model (probably should version it or something)
local_path = os.path.realpath("serialized_models")
model_path = os.path.join(local_path, 'LogisticRegression')
os.mkdir(model_path)

pickle.dump(model, open(model_path + "/logistic_regression.mdl", mode='wb'))
pickle.dump(imputer, open(model_path + "/imputer.imp", mode="wb"))

print("Complete!!!")

