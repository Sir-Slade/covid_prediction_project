# Covid Predictor
The aim of this project is to create a fast, non-invasive and reliable test for determining whether a user is infected with Covid19 or not, using symptoms that can be
easily measured at home such as blood pressure and temprerature. In order to understand how the experiments were performed, I recommend looking at the project report and
the notebooks.

## Project references
- [Coronavirus Disease 2019 (COVID-19) Clinical Data Repository](https://github.com/mdcollab/covidclinicaldata).
- [Project summary and report](https://github.com/Sir-Slade/covid_prediction_project/blob/master/Covid_Predictor_Report.odp)
- [Data Exploration Notebook](https://github.com/Sir-Slade/covid_prediction_project/blob/master/DataExploration.ipynb)
- [Machine learning experiments](https://github.com/Sir-Slade/covid_prediction_project/blob/master/PredictorExperiments.ipynb)

## Quick Deploy

1.  Move the WebApp folder to the directory you want to be hosting the app.
2.  Clone the [covidclinicaldata](https://github.com/mdcollab/covidclinicaldata) project (or move the folder in this github project) to the directory where you want to keep the data.
3.  Create a new python/conda environment and install the following packages (using `conda install <package-name>` or `pip install <package-name>` depending on your environment:
  - `numpy`
  - `pandas`
  - `imblearn`
  - `scikit-learn`
4.  Open a terminal and run the following command to train the model:
  - `python TrainingScript.py <covidclinicaldata location>` (if you dont specify a location, the script will asume 'covidclinicaldata' is in the parent folder.
5.  Execute the following command to run the app locally:
  - `python FlaskApp.py`
6.  Open a browser and go to `http://localhost:5000`
7.  Enjoy the app :)

## Additional deployment notes
It is recommended that you clone the original [covidclinicaldata](https://github.com/mdcollab/covidclinicaldata) project since it might have some new data. (As of 06/09/2021 it doesn't appear to be active though)



