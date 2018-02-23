# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 17:47:45 2018

@author: Andre
"""
import os
import sys
from pathlib import Path
# import typing
from typing import Iterable, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import scipy
import pandas as pd
# import numpy as np
# from nltk.corpus import stopwords

# TODO: Clean up this messy module...
from KaggleWord2VecUtility import KaggleWord2VecUtility


# Whether to run in debug mode.
_DEBUG = True


def debug_print(*objects, sep=' ', end='\n', file=sys.stderr, flush=False):
    """
    If ``_DEBUG`` is set to true, this function behaves just like the builtin
    :py:func:`print` except that it prints to stderr rather than to stdout by
    default.
    """
    if _DEBUG:
        print(*objects, sep=sep, end=end, file=file, flush=flush)


def _get_current_file_dir() -> Path:
    """Returns the directory of the script."""
    try:
        return Path(os.path.realpath(__file__)).parent
    except(NameError):
        return Path('')


# Project root directory, i.e. the github repo directory.
_PROJECT_ROOT = _get_current_file_dir() / '..'

# Where to find the data files.
_DATA_FILE = {
    'labeled':
        _PROJECT_ROOT / 'data' / 'labeledTrainData.tsv',
    'unlabeled':
        _PROJECT_ROOT / 'data' / 'unlabeledTrainData.tsv',
    'test':
        _PROJECT_ROOT / 'data' / 'testData.tsv',
    'result':
        _PROJECT_ROOT / 'results' / 'Prediction.csv'
}


def read_data(keyword: str) -> pd.DataFrame:
    """
    Reads data from the tsv file corresponding to ``_DATA_FILE[keyword]``
    and returns it as a :py:class:`panda.DataFrame`.
    """
    debug_print('[*] Reading ' + keyword + ' data...')
    return pd.read_csv(_DATA_FILE[keyword], header=0, delimiter='\t',
                       quoting=3)


def clean_up_reviews(reviews: Iterable[bytes]):
    """
    Given a list of reviews, strips all HTML markup elements and
    converts the review to a list of lowercase words.
    """
    debug_print("[*] Cleaning and parsing the reviews...")
    return map(
        lambda x: ' '.join(KaggleWord2VecUtility.review_to_wordlist(x, True)),
        reviews)


def learn_vocabulary_and_transform(reviews: Iterable[str]) \
        -> Tuple[scipy.sparse.csr_matrix, CountVectorizer]:
    """
    Given a list of "clean" reviews, converts them to feature vectors
    using a simple Bag of Words method. Returns both feature vectors
    and the vectoriser.
    """
    debug_print('[*] Creating the bag of words...')
    # "CountVectorizer" is scikit-learn's bag of words tool.
    vectorizer = CountVectorizer(analyzer='word', tokenizer=None,
                                 preprocessor=None, stop_words=None,
                                 max_features=5000)
    # fit_transform() does two things. First, it fits the model and
    # learns the vocabulary. Then it transforms the data (i.e. reviews)
    # into feature vectors.
    return vectorizer.fit_transform(reviews), vectorizer


def train_logistic_regression(feauture_vectors, sentiments)\
        -> LogisticRegression:
    """
    Returns a trained Logistic regression on the provided data.
    """
    debug_print('[*] Training the Logistic regression (this may take a while)...')
    logit = LogisticRegression()
    # Fit the forest to the training set, using the bag of words as features
    # and the sentiment labels as the response variable
    return logit.fit(feauture_vectors, sentiments)



def main():
    """
    Uses the Bag of Words method to predict the sentiment labels in the test
    data file.
    """
    train_data = read_data('labeled')
    test_data = read_data('test')
    train_reviews = clean_up_reviews(train_data['review'])
    test_reviews = clean_up_reviews(test_data['review'])

    train_data_features, vectorizer = learn_vocabulary_and_transform(
        train_reviews)
    logit = train_logistic_regression(train_data_features, train_data['sentiment'])
    test_data_features = vectorizer.transform(test_reviews)

    # Use the random forest to make sentiment label predictions
    debug_print('[*] Predicting test labels...')
    prediction = logit.score (test_data_features)
    print(prediction)
    result = logit.predict(test_data_features)


    pd.DataFrame(data={"id": test_data["id"], "sentiment": result}) \
        .to_csv(_DATA_FILE['result'], index=False, quoting=3)
    debug_print('[*] Results\'ve been written to "'
                + str(_DATA_FILE['result']) + '".')
    
if __name__ == '__main__':
    main()