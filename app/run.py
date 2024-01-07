import joblib
import json
import plotly
import pandas as pd
import re 

from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie

from sqlalchemy import create_engine

URL_REGEX = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
STOP_WORDS = set(stopwords.words('english'))

app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../DB/disaster_messages.db')
df = pd.read_sql_table('DB/disaster_messages', engine)

# load model
model = joblib.load("..\models\model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # get frequences with which each class appears in the dataset
    features = df.loc[:,'related':].columns.to_list()
    features.remove('child_alone')
    prevalence_df = df[features].sum() / df.shape[0]
    prevalence_df = prevalence_df.sort_values()

    # get the percentage of english messages in the dataset
    df['english'] = df.apply(lambda x: 'english' if (x['message'] == x['original']) or (x['original'] == None) else 'other language', axis = 1)
    english_percentage = df['english'].value_counts()

    graphs = [
        {
            'data': [
                Bar(
                    x=[f.replace('_', ' ') for f in features],
                    y=prevalence_df.values
                )
            ],

            'layout': {
                'title': 'Prevalence of clases in the entire training dataset',
                'height': 600,
                'yaxis': {
                    'title': "Percentage of all records labeled with a given class"
                },
            }
        },
        {
            'data': [
                Pie(
                    labels=english_percentage.index,
                    values=english_percentage
                )
            ],

            'layout': {
                'title': 'Prevalence of clases in the entire training dataset',
                'yaxis': {
                    'title': "Percentage of all records labeled with a given class"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()