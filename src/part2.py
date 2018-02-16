#!/usr/bin/env python

#  Copyright (C) 2018 Johanna de Vos
#  Copyright (C) 2014-2018 Angela Chapman
#
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Parts 2 and 3 of the tutorial, which cover how to
#  train a model using Word2Vec.
#
# *************************************** #

# Import libraries
import logging
import os
from pathlib import Path
import sys

from gensim.models import Word2Vec
import nltk.data
# from nltk.corpus import stopwords
# import numpy as np
import pandas as pd
# from sklearn.ensemble import RandomForestClassifier

from KaggleWord2VecUtility import KaggleWord2VecUtility


# ****** Read the two training sets and the test set

# Whether to run in debug mode.
_DEBUG = True


def debug_print(*objects, sep='', end='\n', file=sys.stderr, flush=False):
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


# Project root directory, i.e. the Github repo directory.
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


def get_clean_reviews(reviews):
    return map(
        lambda x: KaggleWord2VecUtility.review_to_wordlist(
            x, remove_stopwords=True),
        reviews['review'])


# def get_avg_feature_vecs(reviews, model, num_features):
#     # Given a set of reviews (each one a list of words), calculate
#     # the average feature vector for each one and return a 2D numpy array
#
#     number_reviews = len(reviews)
#     # Preallocate a 2D numpy array, for speed
#     reviewFeatureVecs = np.zeros((len(reviews),num_features),
#                                  dtype='float32')
#
#     for (i, review) in enumerate(reviews):
#        # Print a status message every 1000th review
#        if i % 1000 == 0:
#            debug_print('[*] Review {} of {}\r'.format(i, number_reviews))
#
#        # Call the function (defined above) that makes average feature vectors
#        reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model,
#                                                         num_features)
#
#        # Increment the counter
#        counter = counter + 1.
#     return reviewFeatureVecs


def main():
    # Read data from files
    train_data = read_data('labeled')
    test_data = read_data('test')
    unlabeled_train_data = read_data('unlabeled')

    # Verify the number of reviews that were read (100,000 in total)
    print("Read %d labeled train reviews, %d labeled test reviews, "
          "and %d unlabeled reviews\n" % (train_data["review"].size,
                                          test_data["review"].size,
                                          unlabeled_train_data["review"].size))

    # Load the punkt tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # ****** Split the labeled and unlabeled training sets into clean sentences
    sentences = []  # Initialize an empty list of sentences

    print("Parsing sentences from training set")
    for review in train_data["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review,
                                                               tokenizer)

    print("Parsing sentences from unlabeled set")
    for review in unlabeled_train_data["review"]:
        sentences += KaggleWord2VecUtility.review_to_sentences(review,
                                                               tokenizer)

    # ****** Set parameters and train the word2vec model

    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print("Training Word2Vec model...")
    model = Word2Vec(sentences, workers=num_workers,
                     size=num_features, min_count=min_word_count,
                     window=context, sample=downsampling, seed=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model.save(str(_PROJECT_ROOT / 'results' /
                   '300features_40minwords_10context'))

    # model.doesnt_match("man woman child kitchen".split())
    # model.doesnt_match("france england germany berlin".split())
    # model.doesnt_match("paris berlin london austria".split())
    # model.most_similar("man")
    # model.most_similar("queen")
    # model.most_similar("awful")

    # ****** Create average vectors for the training and test sets
    # print("Creating average feature vecs for training reviews")
    # clean_train_data = get_clean_reviews(train_data)
    # trainDataVecs = getAvgFeatureVecs(clean_train_data, model, num_features)

    # print("Creating average feature vecs for test reviews")
    # clean_test_data = get_clean_reviews(test_data)
    # testDataVecs = getAvgFeatureVecs(clean_test_data, model, num_features)

    # ****** Fit a random forest to the training set, then make predictions

    # Fit a random forest to the training data, using 100 trees
    # forest = RandomForestClassifier(n_estimators=100)

    # print("Fitting a random forest to labeled training data...")
    # forest = forest.fit(trainDataVecs, train_data["sentiment"])

    # Test & extract results
    # result = forest.predict(testDataVecs)

    # Write the test results
    # output = pd.DataFrame(data={"id": test_data["id"], "sentiment": result})
    # output.to_csv(_PROJECT_ROOT / 'results' / 'Word2Vec_BagOfWords.csv',
    #               index=False, quoting=3)
    # print("Wrote Word2Vec_BagOfWords.csv")


if __name__ == '__main__':
    main()
