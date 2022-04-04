import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

random_state = 0


def load_data(database_filepath, table_name):
    """
    Method to load data from the database

    Args:
        database_filename (str): name of database to fetch data from
        table_name (str): name of table to fetch data from

    Returns:
        X (series): feature variable (messages)
        y (dataframe): target variables (outputs)
        category_names (list): names of categories
    """
    
    # create database engine
    engine = create_engine('sqlite:///'+database_filepath)
    
    # read from table in database
    df = pd.read_sql_table(table_name, engine)
    
    # split dataset into feature and target variables
    X = df['message']
    Y = df.drop(columns = ['id', 'message', 'original', 'genre'])
    
    # get category_names
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    """
    Method to process the text data into lemmatized without stop words tokens

    Args:
        text: text data to be processed

    Returns:
        list: clean_tokens list with tokens extracted from the processed text data 
    """
    
    # normalize case and remove leading/trailing white space and punctuation
    text = re.sub("\W"," ", text.lower().strip())
    
    # tokenize
    tokens = word_tokenize(text)
    
    # initiate stopword
    stop_words = stopwords.words("english")
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token to lemmatize and remove stopwords  
    clean_tokens = []
    
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok)
            clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    """
    Method to build ML gridsearch pipeline

    Args:
        None

    Returns:
        model: ML gridsearch pipeline
    """
    
    lr_clf = LogisticRegression(random_state = random_state)
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(lr_clf))   
    ])
    
    params = {
    'vect__decode_error' : ['strict', 'ignore', 'replace'],
    'tfidf__norm' : ['l1', 'l2'],
    'classifier__estimator__C' : [0.1, 1, 5, 10],
    'classifier__estimator__max_iter' : [500, 1000, 5000]
    }

    model = GridSearchCV(pipeline, params)
    
    return model



def evaluate_model(model, X_test, y_test, category_names):
    """
    Method to evaluate ML gridsearch pipeline

    Args:
        model: trained ML gridsearch pipeline
        X_test (series): feature test data (messages)
        y_test (dataframe): target test data (outputs)
        category_names (list): names of categories

    Returns:
        None
    """
    # predict on best estimator
    y_pred = model.predict(X_test)

    # classification report
    print(classification_report(y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Method to evaluate ML gridsearch pipeline

    Args:
        model: trained ML gridsearch pipeline
        model_filepath (str): name for model pickling

    Returns:
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        table_name = 'labeledmessages'
        X, Y, category_names = load_data(database_filepath, table_name)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/disastermanagement.db message_lr_classifier.pkl')


if __name__ == '__main__':
    main()