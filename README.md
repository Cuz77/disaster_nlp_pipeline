# Disaster Response Pipeline Project



## Motivation

This project has been created as a part of the Udacity Nanodegree Data Scientist program. 



## Dependencies

Libraries used for this project:
- flask
- joblib
- json
- nltk
- numpy
- pandas
- plotly
- re
- sklearn
- sqlalchemy
- sys
- warnings



## Summary

This repository houses files for NLP ML algorythm that can predict themes for messages related to natural disasters. The project includes an ETL pipeline for datasets provided in csv files, ML pipeline with tuned hyperparameters, and a Flask app that can be used to predict themese for new messages.



## Structure

The directory structure for this project looks like follows:

    disaster_nlp_pipeline/
    ├── DB/
    │   ├── messages.csv
    │   ├── categories.csv
    │   └── DB.db
    ├── development files/
    │   ├── etl_development.ipynb
    │   ├── ml_pipeline.ipynb
    │   └── DB/
    ├── models/
    │   └── model.pkl
    ├── app/
    │   ├── templates/
    │   │   ├── go.html
    │   │   └── master.html
    │   └── run.py
    ├── .gitignore
    ├── process_data.py
    ├── train_classifier.py
    └── README.md


##### Main directories:
- **DB/** folder containing two csv files and SQL database pre-processed with etl.py
- **development files/** this folder houses Jupyter notebooks used in developing the code (not used by the model)
- **models/** a directory with the final model trained with train.py
- **app/** this is where the Flask app sits

##### Main executables:
- **process_data.py** this file runs the ETL pipeline
- **train_classifier.py** this file runs the ML pipeline



## Instructions:


#### Re-train the model

If there's a need to ever retrain the model, it can be done in three steps:

1. Replaced the DB/messages.csv and DB/categories.csv files with new data if necessary

2. Run the etl.py file to prepare the dataset, e.g.:

    `py process_data.py DB/messages.csv DB/categories.csv DB/`

2. Run the train.py file to retrain the model to a given directory, e.g.:

    `py train_classifier.py DB/ models/`


#### Run the app

To see the results and use the model, run the app accordingly:

1. Open the terminal in the disaster_nlp_pipeline/app directory

2. Run the app with:

    `py run.py`

3. Open this port address in your browser: http://192.168.1.8:3000/ 

4. The home page will show a simple overview of the dataset. To use the model to predict classess of any message, paste it in the dialog box on the top and hit the greenb "Classify Message" button. The app will list all classes, with predicted ones highlighted in green.



## Dataset

The dataset consists of two csv files provided Figure Eight:
1. categories - a file with messages ids labaled with 36 themes either present (1) or not (0) in a given message
2. messages - a file with translated messages pertaining to natural disasters response
     
    

## Acknowledgements

The dataset has been provided by Figure Eight as a part of Udacity Nanodegree Data Science course.

The Flask app has been set up with template code provided by Udacity with little customization.