# Disaster Response Pipeline Project



## Motivation

This project has been created as a part of the Udacity Nanodegree Data Scientist program. 



## Dependencies

Libraries used for this project:

- pandas
- matplotlib
- numpy
- sklearn for modeling, testing, and assesing
- wordcloud
- collections



## Summary

TODO.....................



## Structure

The directory structure for this project looks like follows:

    disaster_nlp_pipeline/
    ├── DB/
    │   ├── messages.csv
    │   ├── categories.csv
    │   ├── DB.db
    ├── development files/
    │   ├── etl_development.ipynb
    │   ├── ml_pipeline.ipynb
    │   ├── DB/
    ├── models/
    │   ├── model.pkl
    ├── app/
    │   ├── ff
    │   ├── ff
    ├── .gitignore
    ├── etl.py
    ├── train.py
    └── README.md


TODO.....................


- **DB** folder contains three csv files with the data
- **boston_airbnb_analysis.ipynb** holds the main code for analysis
- **.gitignore** file used mostly to avoid uploading jupiter checkpoints



### Instructions:


#### 1. ETL



#### 2. Re-train the model



#### 3. Run the app

<!-- 1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage -->



## Dataset

The dataset consists of two csv files provided Figure Eight:
1. categories - a file with messages ids labaled with 36 themes either present (1) or not (0) in a given message
2. messages - a file with translated messages pertaining to natural disasters response
    


## Data preparation



TODO.....................


Categorical variables has been transformed accordingly:
- one-hot encoding has been processed for nominal features
- a feature containing list of available amenities has been transformed into a list of unique values before performing one-hot encoding
- missing values have been replaced with 0s in a separate column denoting a lack of given option since lack of information can impact user behaviour

Numerical variables has been transformed accordingly:
- missing values have been replaced with means to not loose data
- 98% of listings do not show square feet so that feature has been dropped
    
    

## Acknowledgements

The dataset has been provided by Figure Eight as a part of Udacity Nanodegree Data Science course.
