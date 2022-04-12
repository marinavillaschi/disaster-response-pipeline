import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')


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