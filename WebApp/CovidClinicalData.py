import pandas as pd
import numpy as np
import glob
import os

def calculate_union(df_list):
    union = df_list[0].index
    for df in df_list:
        union = np.union1d(union, df.index)        
    return union
   
def read_remote_data():
    from GraphQLRetriever import GraphQLRetriever
    get_names_query = '''
    repository(name: \\\"covidclinicaldata\\\", owner: \\\"mdcollab\\\"){
        object(oid: \\\"984490766e9d6c56832c1577575d9a3b1039ab7c\\\"){
            ... on Tree{
                entries{  
                    extension
                    name
                }
            } 
        }
    }
    '''
    
    endpoint = "https://api.github.com/graphql"
    username = "Sir-Slade"
    token = "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"    
    graphQL = GraphQLRetriever(endpoint, username, token)
    
    response_json = graphQL.post_query(get_names_query)
    print(response_json)
    entry_list = response_json["data"]["repository"]["object"]["entries"]
    all_data_paths = []
    for entry in entry_list:
        if entry["extension"] == ".csv":
            # https://raw.githubusercontent.com is an endpoint that gives us the content of the file in raw form, contrary to the github.com endpoint (much like graphql api)
            all_data_paths.append("https://raw.githubusercontent.com/mdcollab/covidclinicaldata/master/data/{0}".format(entry["name"]))
    return all_data_paths

    
def create_pandas_dataset(all_data_available):
    all_data = None #A workaround to declare the all_data variable for use later

    for file in all_data_available:
        df = pd.read_csv(file)    
        print(file, df["covid19_test_results"].value_counts()["Positive"] / len(df["covid19_test_results"]), "Size:", len(df))
        try:
            df["rapid_flu_results"] = df["rapid_flu_results"].astype("object") #Because in 2 files all values are null and because of that pandas changes the column type to float
            if all_data is None:
                all_data = df
            else:

                all_data = pd.merge(all_data, df, how="outer")

        except Exception as e:
            print(file, "could not be merged:", e)
            print(len(df), "rows were left out")

        print("All data size:", len(all_data))
        
    
    return all_data
    
def clean_data(all_data):        
    '''
    Cleans inplace the dataframe that contains all data of most errors
    '''
    #We eliminate outliers using the thresholds in the following lines. This thresholds were obtained by visually examining the data (DataExploration notebook) since some
    # of the distributions are really narrow and the Quartile method for removing outliers would remove a lot of of the instances.
    low_temp = all_data[all_data["temperature"]<34.5]
    high_temp = all_data[all_data["temperature"]>39]
    hi_pulse = all_data[all_data["pulse"]>=140]
    low_sys = all_data[all_data["sys"]<75]
    high_sys = all_data[all_data["sys"]>= 190]
    low_dia = all_data[all_data["dia"]<=50]
    high_dia = all_data[all_data["dia"]>=110]
    low_rr = all_data[all_data["rr"]<7.5]
    high_rr = all_data[all_data["rr"]>35]
    low_sats = all_data[all_data["sats"]<80]
    
    df_list =[low_temp, hi_pulse, low_sys, high_sys, low_dia, high_dia, low_rr, high_rr, low_sats]
    union = calculate_union(df_list)
    
    all_data.drop(index=union, inplace=True)
    
    #We encode the patients by age group
    all_data.loc[all_data.age <= 28, "age"] = 1
    all_data.loc[(all_data.age > 28) & (all_data.age <= 37), "age"] = 2
    all_data.loc[(all_data.age > 37) & (all_data.age <= 50), "age"] = 3
    all_data.loc[all_data.age > 50, "age"] = 4
    
    #Endoding and handling non-sensical data (having cough severity when you don't have a cough and viceversa) for the cough features
    all_data.loc[((all_data["cough"].isna()) | (all_data["cough"] == False)) & (all_data["cough_severity"].notna()), "cough"] = True
    all_data.loc[(all_data["cough"] == True) & (all_data["cough_severity"].isna()), "cough_severity"] = all_data["cough_severity"].mode()[0] #Since it returns a series, the 0 subscript is to retrieve the value
    all_data.loc[(all_data["cough"] == False) & (all_data["cough_severity"].isna()), "cough_severity"] = 0
    all_data.loc[all_data["cough_severity"] == 'Mild', "cough_severity"] = 1
    all_data.loc[all_data["cough_severity"] == 'Moderate', "cough_severity"] = 2
    all_data.loc[all_data["cough_severity"] == 'Severe', "cough_severity"] = 3    
    
    #Encoding and handling non-sensical data (having shortedness of breath severity when you don't have shortedness of breath and viceversa) for the sob features
    all_data.loc[((all_data["sob"].isna()) | (all_data["sob"] == False)) & (all_data["sob_severity"].notna()), "sob"] = True    
    all_data.loc[(all_data["sob"] == True) & (all_data["sob_severity"].isna()), "sob_severity"] = all_data["sob_severity"].mode()[0] #Since it returns a series, the 0 subscript is to retrieve the value
    all_data.loc[(all_data["sob"] == False) & (all_data["sob_severity"].isna()), "sob_severity"] = 0   
    all_data.loc[all_data["sob_severity"] == 'Mild', "sob_severity"] = 1
    all_data.loc[all_data["sob_severity"] == 'Moderate', "sob_severity"] = 2
    all_data.loc[all_data["sob_severity"] == 'Severe', "sob_severity"] = 3
    
    
    #Handling non-sensical data for fever (some revisiting of the notebook suggests this might have been the wrong approach because the proportions change too much, however
    # it might have been the right call given that fever is one of the most powerful features in the model experiments)
    all_data.loc[(all_data["fever"] == True) & (all_data["temperature"] <37), "fever"] = False
    all_data.loc[(all_data["fever"] == False) & (all_data["temperature"] >=37), "fever"] = True
    
    #These columns have been dropped for several reasons. For more info, check the DataExploration and PredictorExperiments notebooks
    columns_to_drop = ["batch_date", "test_name", "swab_type", "er_referral","rapid_flu_results", "rapid_strep_results", "cxr_findings", "cxr_impression", "cxr_label", "cxr_link", "copd"]
    all_data.drop(columns=columns_to_drop, inplace = True)
    
    #For this particular dataset, the "Severe" level of sob_severity only appeared in negative values, and as such, could confuse the predictor so we drop it
    #However if new data from this source becomes available (someone updates the covidclincaldata repo), the following 2 lines should be commented out or removed
    sob_severe_indexes = all_data[all_data["sob_severity"] == 3].index
    all_data.drop(index=sob_severe_indexes, inplace=True)
    
    all_data.drop_duplicates(subset=all_data.columns.difference(["covid19_test_results"]), inplace=True) #Drop all duplicates that exist between the classes
    
def create_file():
    data = read_data()
    clean_data(data)
    data.to_csv('covidclinicaldata-cleaned.csv', index=False)
    
def calculate_pam(data_x):
    return data_x["dia"] + ((data_x["sys"] - data_x["dia"])/3)
    
class DataImputer():
    '''
    The data imputer designed for this project. It works with any subset of features of the data as well as with with the whole dataset, imputing accordingly.
    It can be configured to calculate the value we want to impute for each feature manually (that is why it is used in favor of the Sklearn ColumnTransformer classes)
    It also takes care of standardization of the vitals.
    '''
    vitals = ["temperature", "pulse", "rr", "sats", "pam", "sys", "dia"]
    
    def __init__(self):
        self.column_values = {}
        self.vitals_values = {}
        
    def fit_transform(self, data_x, data_y):
        self.fit(data_x, data_y)
        return self.transform(data_x, data_y)
        
    def fit(self, data_x, data_y=None):
        self.get_high_risk_exposure_value(data_x, data_y)
        self.get_vitals_values(data_x)
        self.get_a_symptoms_values(data_x)
        self.get_r_symptoms_values(data_x)
        
    def transform(self, data_x, data_y=None):
        imp_data = data_x.copy()      
        
        if "sys" in data_x.columns and "dia" in data_x.columns:
            imp_data["pam"] = calculate_pam(data_x)
            imp_data.drop(columns=["sys", "dia"], inplace=True)
        for feature in imp_data.columns:           
            
            if feature in self.column_values:
                new_value = self.column_values[feature]
                
                if feature == "high_risk_exposure_occupation" and data_y is None:
                    new_value=True            
                    
                if feature == "days_since_symptom_onset" and data_y is not None: #If we are training, we impute depending on the class that we have since it looks like the means for the feature in each class are quite different
                    imp_data.loc[(data_x[feature].isna())  & (data_y == "Positive"), feature] = data_x.loc[(data_y == "Positive"), feature].mean()
                    imp_data.loc[(data_x[feature].isna())  & (data_y == "Negative"), feature] = data_x.loc[(data_y == "Negative"), feature].mean()
                    
                imp_data.loc[imp_data[feature].isna(), feature] = new_value
                
                #We standardize the vitals 
                if feature in DataImputer.vitals:
                    imp_data[feature] = (imp_data[feature] - self.vitals_values[feature][0]) / self.vitals_values[feature][1] 
                    
                #Convert all categorical values to boolean to avoid errors with some models
                if imp_data[feature].dtype == "object": 
                    imp_data[feature] = imp_data[feature].astype("bool")
                    
                
        if "high_risk_interactions" in data_x.columns: #Because this depends on 'high_risk_exposure_occupation being imputed first'
            imp_data.loc[imp_data["high_risk_interactions"].isna(), "high_risk_interactions"] = imp_data["high_risk_exposure_occupation"]   
            imp_data["high_risk_interactions"] = imp_data["high_risk_interactions"].astype("bool")
        
        
        return imp_data
        
    def get_high_risk_exposure_value(self, data_x, data_y):
        if "high_risk_exposure_occupation" in data_x.columns:
            self.column_values["high_risk_exposure_occupation"] = data_x[data_y == "Positive"].high_risk_exposure_occupation.mode()[0]

        
    def get_vitals_values(self, data_x):
        if "temperature" in data_x.columns:
            self.column_values["temperature"] =  data_x["temperature"].mean()
            self.vitals_values["temperature"] = (data_x["temperature"].mean(), data_x["temperature"].std())
            
        if "pulse" in data_x.columns:
            self.column_values["pulse"] = data_x["pulse"].median()
            self.vitals_values["pulse"] = (data_x["pulse"].mean(), data_x["pulse"].std())
            
        if "rr" in data_x.columns:
            self.column_values["rr"] = data_x["rr"].median()
            self.vitals_values["rr"] = (data_x["rr"].mean(), data_x["rr"].std())
            
        if "sats" in data_x.columns:
            self.column_values["sats"] = data_x["sats"].median()
            self.vitals_values["sats"] = (data_x["sats"].mean(), data_x["sats"].std())
            
        if "sys" in data_x.columns and "dia" in data_x.columns:
            pam = calculate_pam(data_x)
            self.column_values["pam"] =  pam.mean()
            self.vitals_values["pam"] = (pam.mean(), pam.std())            
            
        if "pam" in data_x.columns: #if for some reason we already have the pam we do the same as the previous step
            self.column_values["pam"] =  data_x["pam"].mean()
            self.vitals_values["pam"] = (data_x["pam"].mean(), data_x["pam"].std())
            
    def get_a_symptoms_values(self, data_x):
        
        if "ctab" in data_x.columns:
            self.column_values["ctab"] = data_x["ctab"].mode()[0]
            
        if "labored_respiration" in data_x.columns:
            self.column_values["labored_respiration"] = data_x["labored_respiration"].mode()[0]
            
        if "rhonchi" in data_x.columns:
            self.column_values["rhonchi"] = data_x["rhonchi"].mode()[0]
            
        if "wheezes" in data_x.columns:
            self.column_values["wheezes"] = data_x["wheezes"].mode()[0]
            
        if "days_since_symptom_onset" in data_x.columns:
            self.column_values["days_since_symptom_onset"] = data_x["days_since_symptom_onset"].median()
            
    def get_r_symptoms_values(self, data_x):
        if "cough" in data_x.columns:
            self.column_values["cough"] = data_x["cough"].mode()[0]
            
        if "cough_severity" in data_x.columns:
            self.column_values["cough_severity"] = data_x["cough_severity"].mode()[0]
            
        if "fever" in data_x.columns:
            self.column_values["fever"] = data_x["fever"].mode()[0]
            
        if "sob" in data_x.columns:
            self.column_values["sob"] = data_x["sob"].mode()[0]
            
        if "sob_severity" in data_x.columns:
            self.column_values["sob_severity"] = data_x["sob_severity"].mode()[0]
        
        if "diarrhea" in data_x.columns:
            self.column_values["diarrhea"] = data_x["diarrhea"].mode()[0]
            
        if "fatigue" in data_x.columns:
            self.column_values["fatigue"] = data_x["fatigue"].mode()[0]
            
        if "headache" in data_x.columns:
            self.column_values["headache"] = data_x["headache"].mode()[0]
            
        if "loss_of_smell" in data_x.columns:
            self.column_values["loss_of_smell"] = data_x["loss_of_smell"].mode()[0]
            
        if "loss_of_taste" in data_x.columns:
            self.column_values["loss_of_taste"] = data_x["loss_of_taste"].mode()[0]
    
        if "runny_nose" in data_x.columns:
            self.column_values["runny_nose"] = data_x["runny_nose"].mode()[0]
            
        if "muscle_sore" in data_x.columns:
            self.column_values["muscle_sore"] = data_x["muscle_sore"].mode()[0]
            
        if "sore_throat" in data_x.columns:
            self.column_values["sore_throat"] = data_x["sore_throat"].mode()[0]