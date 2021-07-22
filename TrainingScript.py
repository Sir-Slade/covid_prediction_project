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
pam = training_data["dia"] + ((training_data["sys"] - training_data["dia"])/3)
training_data = pd.concat([training_data, pam], axis=1)
training_data.rename({ 0 : "pam"}, axis=1,inplace =True) #To rename the column that was appended (which has a value of 0 that is not a string)

selected_features = ["wheezes", "ctab", "pam", "rr", "pulse", "temperature", "high_risk_exposure_occupation", "cough", "loss_of_smell", "muscle_sore", "loss_of_taste", "headache", "days_since_symptom_onset", "fatigue", "fever", "diabetes"]

#Imputation
imputer = DataImputer()
X = training_data[selected_features]
y = training_data.iloc[:,0]
imputer.fit_transform(X, y)

#Resampling
smote = SMOTE()
X_oversample, y_oversample = smote.fit_resample(X, y)

#Training
model = LogisticRegression()
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

