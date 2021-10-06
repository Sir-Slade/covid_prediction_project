# Covid Predictor
The aim of this project is to create a fast, non-invasive and reliable test for determining whether a user is infected with Covid19 or not, using symptoms that can be
easily measured at home such as blood pressure and temprerature. The result of the experiments done with this project show that a Logistic Regression model with a Newton solver, trained with 1-to-1 oversampled data and using a classification threshold of 0.72 is the model with the highest F2 score at 0.266904, showing that the model is very good at recognizing Negative examples, and at least better at recognizing Positive examples than random guessing, which makes it a viable alternative for identifying a patient that is potentially infected with Covid19. For more detailed information, please check the project summary and report presentation.

The deployed app can be found at https://covid-predictor-project.herokuapp.com/

## Project references
- [Coronavirus Disease 2019 (COVID-19) Clinical Data Repository](https://github.com/mdcollab/covidclinicaldata): The original repo that contains the data I used to build my model.
- [Project summary and report](https://github.com/Sir-Slade/covid_prediction_project/blob/master/Covid_Predictor_Report.odp): A presentation that contains a summary of the process of how the model was developed, as well as final performance metrics for the model.
- [Data Exploration Notebook](https://github.com/Sir-Slade/covid_prediction_project/blob/master/DataExploration.ipynb): Notebook I used for exploring the data and the cleaning process for the experiments.
- [Model selection and Feature engineering Notebook](https://github.com/Sir-Slade/covid_prediction_project/blob/master/PredictorExperiments.ipynb): Notebook I used for the experiments with different models (as described in the project summary presentation).

## Quick Deploy

1.  Move the WebApp folder to the directory you want to be hosting the app.
2.  Clone the [covidclinicaldata](https://github.com/mdcollab/covidclinicaldata) project (or move the folder in this github project) to the directory where you want to keep the data.
3.  Create a new python/conda environment and install the following packages (using `conda install <package-name>` or `pip install <package-name>` depending on your environment:
  - `numpy`
  - `pandas`
  - `imblearn`
  - `scikit-learn`
  - `flask`
4.  Open a terminal and run the following command to train the model:
  - `python TrainingScript.py <covidclinicaldata location>` (if you dont specify a location, the script will asume 'covidclinicaldata' is in the parent folder.
5.  Execute the following command to run the app locally:
  - `python FlaskApp.py`
6.  Open a browser and go to `http://localhost:5000`
7.  Enjoy the app :)

## Additional deployment notes
It is recommended that you clone the original [covidclinicaldata](https://github.com/mdcollab/covidclinicaldata) project since it might have some new data. (As of 06/09/2021 it doesn't appear to be active though)




