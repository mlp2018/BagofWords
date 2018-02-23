#!/usr/bin/env python3

# Copyright (C) 2018 Tom Westerhout
# Copyright (C) 2014-2018 Angela Chapman
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
from pathlib import Path
from typing import Iterable, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import scipy
import pandas as pd
import numpy as np
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
        _PROJECT_ROOT / 'results' / 'Prediction.csv',
    'result_false':
        _PROJECT_ROOT / 'results' / 'FalsePrediction.csv'
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


def train_random_forest(feature_vectors, sentiments) \
        -> RandomForestClassifier:
    """
    Returns a Random Forest trained on the provided data.
    """
    debug_print('[*] Training the random forest (this may take a while)...')
    forest = RandomForestClassifier(n_estimators=100)
    # Fit the forest to the training set, using the bag of words as features
    # and the sentiment labels as the response variable
    return forest.fit(feature_vectors, sentiments)

def evaluate_result(y_hat,y):
    """
    Given some predictions (nx2 array containing ids and predicted label), identifiers and known labels,
    return accuracy of prediction and return list of wrongly classified examples
    """

    incorrect_idx = np.where(y != y_hat)

    acc = (len(y)-len(incorrect_idx[0]))/len(y)

    return acc,incorrect_idx


def main():
    """
    Uses the Bag of Words method to predict the sentiment labels in the test
    data file.
    """
    train_data = read_data('labeled')
    test_data = read_data('test')
    train_reviews = list(clean_up_reviews(train_data['review']))
    #test_reviews = clean_up_reviews(test_data['review'])

    skf = StratifiedKFold(n_splits=3, random_state=None, shuffle=True)
    results = [[] for y in range(2)]
    for idx, skf in enumerate(skf.split(train_reviews,train_data["sentiment"])):
        debug_print('Processing fold number',idx+1)
        train_index = skf[0]
        test_index = skf[1]
        split_train_reviews = [train_reviews[i] for i in train_index]
        split_test_reviews = [train_reviews[i] for i in test_index]

        train_data_features, vectorizer = learn_vocabulary_and_transform(split_train_reviews)
        forest = train_random_forest(train_data_features, train_data["sentiment"][train_index])
        test_data_features = vectorizer.transform(split_test_reviews)

        # Use the random forest to make sentiment label predictions
        debug_print('[*] Predicting test labels...')#implement counter and print fold number
        results[0].extend(train_data["id"][test_index])
        results[1].extend(forest.predict(test_data_features))

    #results are in shuffeled order. sort ids and labels accordingly for both known and predicted labels
    sorted_labels = train_data["sentiment"][np.argsort(train_data["id"])]
    sorted_reviews = train_data["review"][np.argsort(train_data["id"])]
    sorted_prediction = [results[1][i] for i in np.argsort(results[0])]

    acc, idx = evaluate_result(sorted_labels, sorted_prediction)

    debug_print('Overall accuracy on left-out data was',acc)

    #save wrongly classified predictions for later inspection
    pd.DataFrame(data={"review": sorted_reviews[idx[0]]}).to_csv(_DATA_FILE['result_false'], index=False, quoting=3, escapechar='\\')

    pd.DataFrame(data={"id": results[0], "sentiment": results[1]}) \
     .to_csv(_DATA_FILE['result'], index=False, quoting=3)
    debug_print('[*] Results\'ve been written to "'
        + str(_DATA_FILE['result']) + '".')


if __name__ == '__main__':
    main()
