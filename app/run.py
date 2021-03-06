import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar, Histogram
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/disastermanagement.db')
df = pd.read_sql_table('labeledmessages', engine)

# load model
model = joblib.load("../models/message_lr_classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    Y = df.drop(columns = ['id', 'message', 'original', 'genre'])
    categories_names = (Y.columns)

    Y['sum'] = Y.sum(axis=1)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=categories_names.str.replace('_', ' '),
                    y=Y.sum().sort_values(ascending=False)[:10]
                )
            ],

            'layout': {
                'title': 'Top 10 categories of messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories",
                    # 'tickangle' : "0"
                }
            }
        },
        {
            'data': [
                Histogram(
                    x=Y['sum']
                )
            ],

            'layout': {
                'title': 'Histogram of number of categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Number of categories",
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
    app.run(host='0.0.0.0', port=3003, debug=True)


if __name__ == '__main__':
    main()