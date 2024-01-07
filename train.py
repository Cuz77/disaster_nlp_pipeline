# import libraries
import re
import sys
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import joblib

nltk.download('wordnet')
nltk.download('punkt') 
nltk.download('stopwords') 

from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize, punkt
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from sqlalchemy import create_engine


# Prevent sklearn from printing ConvergenceWarning (due to max iterations limit)
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category = ConvergenceWarning)

# Define static values to detect hyperlinks and remove stop words
URL_REGEX = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
STOP_WORDS = set(stopwords.words('english'))



def load_data(dataset_path):
    """
    Desc: Loads a dataset
    
        Parameters:
            dataset_path (str): a directory path where the csv files are stores
        Returns:
            X (obj): Pandas Series object with text feature
            y (obj): Pandas DataFrame object with target classes 
    """
    # load datasets
    dataset_path = dataset_path if dataset_path[-1] == '/' else dataset_path + '/' # make sure to include trailing /
    engine = create_engine(f'sqlite:///{dataset_path}disaster_messages.db')
    df = pd.read_sql(f'{dataset_path}disaster_messages', con=engine)

    # define X and y
    X = df['message']
    y = df[features]

    return X, y


def tokenize(text):
    """
    Desc: Returns cleaned and lemmatized tokens from a text to be used by an NLP vectorizer

        Parameters:
            text (str): a document to be processed (e.g. a twitter message)
        Returns:
            clean_tokens (list[str]): a list of cleaned and lemmatized word tokens
    """
    # find and replace all hyperlinks 
    urls = re.findall(URL_REGEX, text)

    for url in urls:
        text = text.replace(url, '<url>')

    # tokenize
    tokens = word_tokenize(text)

    # lemmatize and clean words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    clean_tokens = [tok for tok in clean_tokens if tok not in STOP_WORDS]

    return clean_tokens
    
    
def build_model(X, y):
    """
    Desc: Builds the NLP model

        Parameters:
            X (obj): Pandas Series object with text feature
            y (obj): Pandas DataFrame object with target classes 
        Returns:
            cv (obj): GridSearchCV object with parameters to be tuned
    """
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(estimator=SGDClassifier()))
               ])
    
    # define parameters for GridSearchCV
    parameters = {'clf__estimator__penalty' : ['l1', 'l2', 'elasticnet'],
                  'clf__estimator__loss': ['hinge', 'log_loss', 'squared_hinge', 'perceptron'],
                  'clf__estimator__max_iter' : [200, 500, 1000]
                  }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
        
    return cv


def train_model(X, y, model):
    """
    Desc: Trains the NLP model

        Parameters:
            X (obj): Pandas Series object with text feature
            y (obj): Pandas DataFrame object with target classes 
            model (obj): GridSearchCV object with parameters to be tuned
        Returns:
            best_clf (obj): Trained model object
    """
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8)
    
    # fit model
    best_clf = model.fit(X_train, y_train)
    print(f'Best parameters: {best_clf.best_params_}\n')
    
    # output model test results
    y_pred = best_clf.predict(X_test)
    pred_df = pd.DataFrame(y_pred, columns=y_test.columns)
    report_df = pd.DataFrame(columns=['precision', 'recall', 'f1-score'])

    for col in pred_df.columns:
        scores = classification_report(y_test[col], pred_df[col], output_dict=True, zero_division=0)['weighted avg']
        precision, recall, f1_score, _ = [score for score in scores.values()]
        report_df.loc[len(report_df)] = [precision, recall, f1_score]

    report_df.index = pred_df.columns
    print('Success:\n', report_df.mean(), '\n\n')
    print('Detailed report:\n',report_df, '\n')
    
    return best_clf
    

def export_model(model):
    """
    Desc: Saves given trained model to 'models\model.pkl'

        Parameters:
            best_clf (obj): Trained model object
    """
    joblib.dump(model, 'models\model.pkl')


def run_pipeline():
    """
    Desc: Runs the complete ML pipeline from loading the dataset, through tuning, to saving final model
    """
    if len(sys.argv) == 2:
        dataset_path = sys.argv[1]
        
        print('Loading data...\n')
        X, y = load_data(dataset_path)
        
        print('Building model...\n')
        model = build_model(X, y)

        print('Training model...\n')
        model = train_model(X, y, model)
        
        print('Exporting model...\n')
        export_model(model)
        
        print('All done.')
        
    else:
        print('This function requires one positional argument:\n-a directory relative path to the disaster_messages.db, e.g. "database/"')
        

    
if __name__ == '__main__':
    run_pipeline()