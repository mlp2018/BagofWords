#!/usr/bin/env python

#  Copyright (C) 2018 Johanna de Vos
#  Copyright (C) 2014-2018 Angela Chapman
#
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Part 3 of the tutorial and covers Bag of Centroids
#  for a Word2Vec model. This code assumes that you have already
#  run Word2Vec and saved a model called "300features_40minwords_10context"
#
# *************************************** #


# Import libraries
import numpy as np
import os
import pandas as pd
import re
import sys
import time
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from KaggleWord2VecUtility import KaggleWord2VecUtility
from nltk.corpus import stopwords
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.naive_bayes import BernoulliNB


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
    'labeled' :
        _PROJECT_ROOT / 'data' / 'labeledTrainData.tsv',
    'unlabeled' :
        _PROJECT_ROOT / 'data' / 'unlabeledTrainData.tsv',
    'test' :
        _PROJECT_ROOT / 'data' / 'testData.tsv',
    'result' :
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
        reviews)

# ****** Define functions to create average word vectors

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given paragraph

    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0.

    # Index2word is a list that contains the names of the words in the model's vocabulary.
    # Convert it to a set, for speed.
    index2word_set = set(model.wv.index2word)

    # Loop over each word in the review and, if it is in the model's vocaublary,
    # add its feature vector to the total.
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])

    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array

    # Initialize a counter
    counter = 0.

    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features), dtype="float32")

    # Loop through the reviews
    for review in reviews:

       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print("Review %d of %d" % (counter, len(reviews)))

       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, \
           num_features)

       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs


# Define a function to create bags of centroids

def create_bag_of_centroids(wordlist, word_centroid_map):
    # The number of clusters is equal to the highest cluster index in the word / centroid map
    num_centroids = max(word_centroid_map.values()) + 1

    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")

    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1

    # Return the "bag of centroids"
    return bag_of_centroids


def main():
    # Load the model
    model = Word2Vec.load(str(_PROJECT_ROOT / 'results' /
                              '300features_40minwords_10context'))

    # Explore the model
    # type(model.wv.syn0)
    # model.wv.syn0.shape
    # model["flower"]

    # Create clean_train_reviews and clean_test_reviews as we did before

    # Read data from files
    train_data = read_data('labeled')
    test_data = read_data('test')

    clean_train_reviews = list(get_clean_reviews(train_data['review']))
    clean_test_reviews = list(get_clean_reviews(test_data['review']))

    # ****************************************************************
    # Calculate average feature vectors for training and testing sets,
    # using the functions we defined above. Notice that we now use stop word removal.

    num_features = 300    # Word vector dimensionality

    debug_print("[*] Creating average feature vecs for training data")
    trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

    debug_print("[*] Creating average feature vecs for testing data")
    testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

    # Fit a Multi-variate Bernoulli Naive Bayes  to the training data, using .1 smoothing parameter
    bayes = BernoulliNB(alpha = 0.1)

    debug_print("[*] Fitting a Multi-variate Bernoulli Naive Bayes to labeled training data...")
    bayes = bayes.fit(trainDataVecs, train_data["sentiment"])

    # Test & extract results
    result = bayes.predict(testDataVecs)

    # Write the test results
    output = pd.DataFrame(data={"id":test_data["id"], "sentiment":result})
    output.to_csv(_PROJECT_ROOT / 'results' / 'Word2Vec_AverageVectors.csv',
                  index=False, quoting=3)

    # ****** Run k-means on the word vectors and print a few clusters
    # This takes +- 29 minutes on Johanna's laptop.
    start = time.time() # Start time

    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an average of 5 words per cluster
    word_vectors = model.wv.syn0
    num_clusters = int(word_vectors.shape[0] / 5)

    # Initalize a k-means object and use it to extract centroids
    debug_print("[*] Running K means")
    kmeans_clustering = KMeans(n_clusters = num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    debug_print("[*] Time taken for K Means clustering: ", elapsed, "seconds.")

    # Create a Word / Index dictionary, mapping each vocabulary word to a cluster number
    word_centroid_map = dict(zip(model.wv.index2word, idx))

    # Print the first ten clusters
    for cluster in range(0,10):

        # Print the cluster number
        debug_print("[*] \nCluster %d" % cluster)

        # Find all of the words for that cluster number, and print them out
        words = []
        for i in range(0,len(word_centroid_map.values())):
            if list(word_centroid_map.values())[i] == cluster:
                words.append(list(word_centroid_map.keys())[i])
        print(words)

    # ****** Create bags of centroids
    # Pre-allocate an array for the training set bags of centroids (for speed)
    train_centroids = np.zeros((train_data["review"].size, num_clusters), dtype="float32")

    # Transform the training set reviews into bags of centroids
    counter = 0
    for review in clean_train_reviews:
        train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    # Repeat for test reviews
    test_centroids = np.zeros((test_data["review"].size, num_clusters), dtype="float32")

    counter = 0
    for review in clean_test_reviews:
        test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
        counter += 1

    # ****** Fit a Multi-variate Bernoulli Naive Bayes and extract predictions
    bayes = BernoulliNB(alpha=0.1)

    # Fitting the Multi-variate Bernoulli Naive Bayes  may take a few minutes
    debug_print("[*] Fitting a Multi-variate Bernoulli Naive Bayes to labeled training data...")
    bayes = bayes.fit(train_centroids, train_data["sentiment"])
    result = bayes.predict(test_centroids)

    # Write the test results
    output = pd.DataFrame(data={"id":test_data["id"], "sentiment":result})
    output.to_csv(_PROJECT_ROOT / 'results' / 'Word2Vec_BagOfCentroids.csv',
                  index=False, quoting=3)
    debug_print("[*] Wrote Word2Vec_BagOfCentroids.csv")


if __name__ == '__main__':
    main()
