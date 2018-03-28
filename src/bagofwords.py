#!/usr/bin/env python3

# Copyright (C) 2018 Sophie Arana
# Copyright (C) 2018 Johanna de Vos
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

import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Type, Any, Iterable, Optional, \
                   Tuple, List, Dict, Callable
import warnings

from bs4 import BeautifulSoup
from gensim.models import Word2Vec, KeyedVectors
import nltk.corpus
import nltk.tokenize
import numpy as np
import pandas as pd
# from scipy.sparse import csr_matrix
import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
# from sklearn.feature_extraction.text import VectorizerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
import tensorflow as tf

import cProfile


def _get_sklearn_version() -> Tuple[int, int, int]:
    """
    Returns the version of scikit-learn as a tuple.
    """
    (a, b, c) = sklearn.__version__.split('.')
    return int(a), int(b), int(c)


if _get_sklearn_version() >= (0, 19, 0):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import StratifiedShuffleSplit
else:  # We have an old version of sklearn...
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.cross_validation import StratifiedShuffleSplit

import gensim


def _get_current_file_dir() -> Path:
    """Returns the directory of the script."""
    try:
        return Path(os.path.realpath(__file__)).parent
    except(NameError):
        return Path(os.getcwd())


# Project root directory, i.e. the github repo directory.
_PROJECT_ROOT = _get_current_file_dir() / '..'

# Default configuration options.
# WARNING: Please, avoid changing it. Use a local `conf.py` in the project's
# root directory.
_DEFAULT_CONFIG = {
    # Options that specify where to read the data from.
    'in': {
        'labeled':   str(_PROJECT_ROOT / 'data' / 'labeledTrainData.tsv'),
        'unlabeled': str(_PROJECT_ROOT / 'data' / 'unlabeledTrainData.tsv'),
        'test':      str(_PROJECT_ROOT / 'data' / 'testData.tsv'),
        'clean':     str(_PROJECT_ROOT / 'data' / 'cleanReviews.tsv'),
    },
    # Options that specify where to write the results to.
    'out': {
        'result':       str(_PROJECT_ROOT / 'results' / 'Prediction.csv'),
        'wrong_result': str(_PROJECT_ROOT / 'results' / 'FalsePrediction.csv'),
    },
    # High-level algorithm specific options.
    'run': {
        # Type of the run, one of {'optimization', 'submission'}
        'type':             'optimization',
        # How many splits to use in the StratifiedKFold
        'number_splits':    3,
        # When preprocessing the reviews, should we remove the stopwords?
        'remove_stopwords': False,
        # Should we cache the preprocessed reviews?
        'cache_clean':      True,
        # After the running the StratifiedKFold on the 90%, should we test the
        # result on the remaining 10?
        'test_10':          False,
        # Random seed used for the 90-10 split.
        'random':           42,
        # How many percent of the data should be left out for testing? I.e.
        # what is "10" in the 90-10 split.
        'alpha':            0.1,
    },
    # Type of the vectorizer, one of {'word2vec', 'bagofwords'}
    'vectorizer': 'word2vec',
    # Type of the classifier to use, one of
    # {'random-forest', 'logistic-regression'}
    'classifier': 'logistic-regression',
    # Options specific to the bagofwords vectorizer.
    'bagofwords': {},
    # Options specific to the word2vec vectorizer.
    'word2vec': {
        # File name where to save/read the model to/from. Dictionary file is
        # computes from it as
        # dictionary_file = model_file + '.dict.npy'
        'model':      str(_PROJECT_ROOT / 'data'
                          / 'GoogleNews-vectors-negative300.bin'),
        # 'model':      str(_PROJECT_ROOT / 'results'
        #                   / '300features_40minwords_10context'),
        # Retrain the model every time?
        'retrain':    False,
        # Averaging strategy to use, one of {'average', 'k-means'}
        'strategy':   'average'
    },
    # Options specific to the random forest classifier.
    'random-forest': {
        'n_estimators': 100,
        'n_jobs':       4,
    },
    # Options specific to the logistic regression classifier.
    'logistic-regression': {
        # TODO: I might've messed this part up. Someone with a good
        # understanding of the required arguments, please check!
        #                                                Tom
        'penalty':           'l2',
        'dual':              True,
        'tol':               0.0001,
        'C':                 1,
        'fit_intercept':     True,
        'intercept_scaling': 1.0,
        'class_weight':      None,
        'random_state':      None,
    },
    # Options specific to the "average" averaging strategy.
    'average': {},
    # Options specific to the "k-means" averaging strategy.
    'k-means': {
        'number_clusters_frac': 0.2,  # NOTE: This argument is required!
        'max_iter':             10,
        'n_jobs':               4,
    },
}


# If you have a local conf.py which defines the configuration options, it will
# be used. Otherwise, this falls back to defaults.
# NOTE: See _DEFAULT_CONFIG for the format of the configuratio options.
try:
    from conf import conf
except ImportError:
    conf = _DEFAULT_CONFIG


# TODO: Fix this.
# Turn off warnings about Beautiful Soup (Johanna has checked all of them
# manually)
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


def _read_data_from(path: str) -> pd.DataFrame:
    assert type(path) == str
    logging.info('Reading data from {!r}...'.format(path))
    return pd.read_csv(path, header=0, delimiter='\t', quoting=3)


class ReviewPreprocessor(object):
    """
    :py:class:`ReviewPreprocessor` is an utility class for processing raw HTML
    text into segments for further learning.
    """

    _stopwords = set(nltk.corpus.stopwords.words('english'))
    _tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    @staticmethod
    def _striphtml(text: str) -> str:
        assert type(text) == str

        text = re.sub('(www.|http[s]?:\/)(?:[a-zA-Z]|[0-9]|[$-_@.&+]'
                      '|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                      '', text)
        text = BeautifulSoup(text, 'html.parser').get_text()
        return text

    @staticmethod
    def _2wordlist(text: str, remove_stopwords: bool = False) -> List[str]:
        assert type(text) == str
        assert type(remove_stopwords) == bool

        text = re.sub('[^a-zA-Z]', ' ', text)
        words = text.lower().split()
        if remove_stopwords:
            words = [w for w in words if w not in
                     ReviewPreprocessor._stopwords]
        assert type(words) == list
        return words

    @staticmethod
    def review2wordlist(review: str, remove_stopwords: bool = False) \
            -> List[str]:
        """
        Given a review, parses it as HTML, removes all URLs and
        non-alphabetical characters, and optionally removes the stopwords. The
        review is then split into lowercase words and the resulting list is
        returned.

        :param str review:            The review as raw HTML.
        :param bool remove_stopwords: Whether to remove the stopwords.
        :return:                      Review split into words.
        :rtype:                       List[str]
        """
        assert type(review) == str
        assert type(remove_stopwords) == bool

        return ReviewPreprocessor._2wordlist(
            ReviewPreprocessor._striphtml(review),
            remove_stopwords=remove_stopwords)

    @staticmethod
    def review2sentences(review: str, remove_stopwords: bool = False) \
            -> Iterable[List[str]]:
        """
        Given a review, splits it into sentences where each sentence
        is a list of words.

        :param str review:            The review as raw HTML.
        :param bool remove_stopwords: Whether to remove the stopwords.
        :return:                      Review split into sentences.
        :rtype:                       Iterable[List[str]].
        """
        assert type(review) == str
        assert type(remove_stopwords) == bool
        raw_sentences = ReviewPreprocessor._tokenizer.tokenize(
            ReviewPreprocessor._striphtml(review))
        return map(
            lambda x: ReviewPreprocessor._2wordlist(x, remove_stopwords),
            filter(lambda x: x, raw_sentences))


# TODO: This function should probably also do the stemming, see
# https://github.com/mlp2018/BagofWords/issues/7.
def clean_up_reviews(reviews: Iterable[str],
                     remove_stopwords: bool = True,
                     clean_file: Optional[str] = None) -> np.ndarray:
    """
    Given an list of reviews, either loads pre-computed clean reviews from
    ``clean_file`` or applies :py:func:`ReviewPreprocessor.review2wordlist` to
    each review and converts it into an :py:class:`numpy.ndarray`. Also, if
    ``clean_file`` is not ``None`` clean reviews are saved to it.

    :param reviews:               Reviews to clean up.
    :type reviews:                Iterable[str]
    :param bool remove_stopwords: Whether to remove the stopwords.
    :param clean_file:            File where clean reviews are stored.
    :type clean_file:             Optional[str]
    :return:                      Iterable of clean reviews.
    :rtype:                       Iterable[str]
    """
    assert isinstance(reviews, Iterable)
    assert type(remove_stopwords) is bool
    assert clean_file is None or type(clean_file) is str
    if clean_file is not None and Path(clean_file).exists():
        return _read_data_from(clean_file)['review'].values
    logging.info('Cleaning and parsing the reviews...')
    review = np.array(list(map(
        lambda x: ' '.join(
            ReviewPreprocessor.review2wordlist(x, remove_stopwords)),
        reviews)), object)
    # This assert is important to ensure we're doing things in a
    # memory-efficient way!
    assert review.dtype == np.object
    if clean_file is not None:
        logging.info('Saving clean data to file {!r}...'
                     .format(repr(clean_file)))
        pd.DataFrame(data={"review": review}) \
            .to_csv(clean_file, index=False, quoting=3)
    return review


def reviews2sentences(reviews: Iterable[str],
                      remove_stopwords: bool = True) -> Iterable[List[str]]:
    """
    Given a list of reviews, cleans them up, splits into sentences, and returns
    a list of *all* the sentences.

    :param reviews:               Reviews to clean up.
    :type reviews:                Iterable[str]
    :param bool remove_stopwords: Whether to remove the stopwords.
    :return:                      Iterable of sentences, where each sentence is
                                  a list of words.
    :rtype:                       Iterable[List[str]]
    """
    logging.info('Splittings reviews into sentences...')
    assert isinstance(reviews, Iterable)
    assert type(remove_stopwords) is bool

    class R2SIter:
        def __init__(self, xs, remove):
            self.reviews = xs
            self.remove_stopwords = remove

        def __iter__(self):
            for review in self.reviews:
                for sentence in ReviewPreprocessor.review2sentences(
                        review, self.remove_stopwords):
                    yield sentence
    return R2SIter(reviews, remove_stopwords)


def submission_run(reviews: Type[np.ndarray],
                   sentiments: Type[np.ndarray],
                   test_reviews: Type[np.ndarray],
                   ids,
                   mk_vectorizer: Callable[[], Any],
                   mk_classifier: Callable[[], Any],
                   prediction_file: str) -> Type[np.array]:
    """
    :param ids: Array of review identifiers.
    :type ids: 'numpy.ndarray' of shape ``(N,)``
    :param reviews: Array of raw reviews texts.
    :type reviews: 'numpy.ndarray' of shape ``(N,)``
    :param sentiments: Array of review sentiments.
    :type sentiments: 'numpy.ndarray' of shape ``(N,)``
    :param test_reviews: Array of test review texts.
    :type test_reviews: 'numpy.ndarray' of shape ``(N,)``
    :param mk_vectorizer: Factory function to create a new vectorizer.
    :type mk_vectorizer: Callable[[], Vectorizer]
    :type mk_classifier: Callable[[], Classifier]
    :param prediction_file
    """

    score, prediction = run_one_fold((reviews, sentiments),
                                     test_reviews,
                                     mk_vectorizer(),
                                     mk_classifier())

    logging.info('Saving all predicted sentiments to {!r}...'
                 .format(prediction_file))
    pd.DataFrame(data={'id': ids, 'sentiment': prediction}) \
        .to_csv(prediction_file, index=False, quoting=3)


def bookkeeping(reviews: Type[np.ndarray],
                wrong_index: Type[np.ndarray],
                wrong_prediction_file: str):
    logging.info('Saving wrong predictions to {!r}...'
                 .format(wrong_prediction_file))
    pd.DataFrame(data={'review': reviews[wrong_index]}) \
        .to_csv(wrong_prediction_file, index=False, quoting=3, escapechar='\\')


def run_one_fold(train_data: Tuple[Type[np.ndarray], Type[np.ndarray]],
                 test_data: Tuple[Type[np.ndarray], Type[np.ndarray]],
                 vectorizer: Any, classifier: Any) \
        -> Tuple[float, Type[np.ndarray]]:
    """
    Given some data to train on and some data to test on, runs the whole
    feature extraction + classification procedure, computes the ROC AUC score
    of the predictions and returns it along with raw predictions.

    :param train_data: ``(array of reviews, array of sentiments)`` on which the
                       model will be trained.
    :param test_data:  ``(array of reviews, array of sentiments)`` on which the
                       accuracy of the model will be computed.
    :param vectorizer: Vectorizer to use.
    :param classifier: Classifier to use.
    :return:           ``(score, predictions)`` tuple.
    """
    score = None
    (train_reviews, train_labels) = train_data
    if isinstance(test_data, tuple):
        (test_reviews, test_labels) = test_data
    else:
        test_reviews = test_data
    logging.info('Transforming training data...')
    train_features = vectorizer.fit_transform(train_reviews)
    logging.info('Transforming test data...')
    test_features = vectorizer.transform(test_reviews)
    logging.info('Fitting...')
    classifier = classifier.fit(train_features, train_labels)
    logging.info('Predicting test labels...')
    prediction = classifier.predict(test_features)
    if isinstance(test_data, tuple):
        score = roc_auc_score(test_labels, prediction)
        logging.info('ROC AUC for this fold is {}.'.format(score))
    return score, prediction


def split_90_10(data: Tuple[Type[np.ndarray], Type[np.ndarray]],
                alpha: float,
                seed: int = 42) \
        -> Tuple[Tuple[Type[np.ndarray], Type[np.ndarray]],
                 Tuple[Type[np.ndarray], Type[np.ndarray]]]:
    """
    Despite the very descriptive name this function does the ``1-alpha`` -
    ``alpha`` split rather than the ``90%`` - ``10%`` one.

    :param data:  The data to split.
    :param alpha: Percentage of data to use for testing.
    :param seed:  Random seed to use for splitting.
    :return: ``((reviews to train on, sentiments to train on),
                (reviews to test on, sentiments to test on))``.
    """
    assert 0 < alpha and alpha < 1
    reviews, labels = data
    train_index = None
    test_index = None
    if _get_sklearn_version() >= (0, 19, 0):
        train_index, test_index = StratifiedShuffleSplit(
            n_splits=1, test_size=alpha,
            random_state=seed).split(reviews, labels).__iter__().__next__()
    else:
        train_index, test_index = StratifiedShuffleSplit(
            labels, n_iter=1, test_size=alpha,
            random_state=seed).__iter__().__next__()
    x_train = reviews[train_index]
    x_test = reviews[test_index]
    y_train = labels[train_index]
    y_test = labels[test_index]
    return (x_train, y_train), (x_test, y_test)


def run_a_couple_of_folds(data: Tuple[Type[np.ndarray], Type[np.ndarray]],
                          mk_vectorizer: Callable[[], Any],
                          mk_classifier: Callable[[], Any],
                          number_splits: int) \
        -> Tuple[float, float, np.ndarray]:
    # TODO: Write docs for this function.

    reviews, labels = data
    predictions = np.zeros(labels.shape, dtype=np.bool_)
    scores = np.zeros((number_splits,), dtype=np.float32)

    def go(idx, train_index, test_index):
        logging.info('Processing fold number {}...'.format(idx + 1))
        scores[idx], predictions[test_index] = run_one_fold(
            (reviews[train_index], labels[train_index]),
            (reviews[test_index], labels[test_index]),
            mk_vectorizer(), mk_classifier())

    if _get_sklearn_version() >= (0, 19, 0):
        skf = StratifiedKFold(n_splits=number_splits, shuffle=False)
        for idx, (train_index, test_index) \
                in enumerate(skf.split(reviews, labels)):
            go(idx, train_index, test_index)
    else:
        skf = StratifiedKFold(labels, n_folds=number_splits, shuffle=False)
        for idx, (train_index, test_index) in enumerate(skf):
            go(idx, train_index, test_index)

    score, score_std = np.mean(scores), np.std(scores)
    logging.info(('Overall accuracy on left-out data from k-fold '
                  'cross-validation was {}').format(score))
    wrong_index = np.where(predictions != labels)
    return score, score_std, wrong_index


# TODO: Split this messy function into smaller ones.
def optimization_run(reviews: Type[np.ndarray],
                     sentiments: Type[np.ndarray],
                     mk_vectorizer: Callable[[], Any],
                     mk_classifier: Callable[[], Any],
                     conf,
                     wrong_prediction_file: str) -> Type[np.array]:
    """
    This is basically the main function.

    :param ids: Array of review identifiers.
    :type ids: 'numpy.ndarray' of shape ``(N,)``
    :param reviews: Array of raw reviews texts.
    :type reviews: 'numpy.ndarray' of shape ``(N,)``
    :param sentiments: Array of review sentiments.
    :type sentiments: 'numpy.ndarray' of shape ``(N,)``
    :param mk_vectorizer: Factory function to create a new vectorizer.
    :type mk_vectorizer: Callable[[], Vectorizer]
    :type mk_classifier: Callable[[], Classifier]
    :param n_splits
    :param test_on_10
    """
    data_90, data_10 = split_90_10((reviews, sentiments),
                                   conf['alpha'], conf['random'])
    reviews_90, _ = data_90
    reviews_10, labels_10 = data_10
    score, sigma, wrong_idx = run_a_couple_of_folds(
        data_90, mk_vectorizer, mk_classifier, conf['number_splits'])
    wrong_reviews = reviews_90[wrong_idx]
    if conf['test_10']:
        logging.info('Training on 90% and testing on 10%...')
        score_10, prediction_10 = run_one_fold(
            data_90, data_10, mk_vectorizer(), mk_classifier())
        wrong_idx_10 = np.where(prediction_10 != labels_10)
        wrong_reviews = np.concatenate(
            (wrong_reviews, reviews_10[wrong_idx_10]), axis=0)
    logging.info('Saving wrong predictions to {!r}...'
                 .format(wrong_prediction_file))
    pd.DataFrame(data={'review': wrong_reviews}) \
        .to_csv(wrong_prediction_file, index=False, quoting=3, escapechar='\\')


class SimpleAverager(object):
    """
    This class implements the simplest strategy for reducing a list of
    word2vec vectors into a single one -- averaging.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _make_avg_feature_vector(
        words: str, known_words: Dict[str, np.ndarray],
            average_vector: Optional[np.ndarray] = None) -> Type[np.ndarray]:
        """
        Given a list of words, returns the their average.

        :param str words:      Words to average.
        :param known_words:    Dictionary mapping known words to vectors.
        :type known_words:     Dict[str, np.ndarray]
        :param average_vector: Pre-allocated zero-initialised vector. It will
                               contain the average of ``words``.
        :type average_vector:  np.ndarray
        :return:               The average of ``words``.
        :rtype:                np.ndarray
        """
        assert type(words) is str
        assert type(known_words) is dict
        assert type(average_vector) is np.ndarray
        word_count = sum(
            1 for _ in map(lambda x: np.add(average_vector, known_words[x],
                                            out=average_vector),
                           filter(lambda x: x in known_words, words.split()))
        )
        return np.divide(average_vector, float(word_count), out=average_vector)

    def transform(self, reviews: np.ndarray, model: Dict[str, np.ndarray]) \
            -> np.ndarray:
        """
        Given a list of reviews and a dictionary of known words, returns an
        array of average feature vectors.

        :param reviews: Reviews to transform.
        :type reviews:  np.ndarray
        :param model:   Dictionary of known words.
        :type model:    Dict[str, np.ndarray]
        :return:        Array of average feature vectors.
        :rtype:         np.ndarray
        """
        assert type(reviews) is np.ndarray
        assert type(model) is dict
        # don't know how to do that properly
        # number_features = len(model['dog'])
        (number_features,) = model.values().__iter__().__next__().shape
        (number_reviews,) = reviews.shape
        feature_vectors = np.zeros(
            (number_reviews, number_features), dtype='float32')
        for (i, (review, vector)) in enumerate(zip(reviews, feature_vectors)):
            if i % 1000 == 0:
                logging.info('PROGRESS: At review #{} of {}...'
                             .format(i, number_reviews))
            SimpleAverager._make_avg_feature_vector(
                review, model, vector)
        return feature_vectors

    def fit_transform(self, reviews: np.ndarray, model: Dict[str, np.ndarray])\
            -> np.ndarray:
        """
        :py:class:`SimpleAverager` has no state, and :py:func:`fit_transform`
        thus simply calls :py:func:`transform`.
        """
        return self.transform(reviews, model)


class KMeansAverager(object):
    """
    This class implements the 'k-means' strategy for reducing a list of
    word2vec vectors into a single one.
    """

    def __init__(self, **kwargs: dict) -> None:
        self.number_clusters_frac = kwargs['number_clusters_frac']
        del kwargs['number_clusters_frac']
        self.kmeans_args = kwargs
        self.kmeans = None
        self.word2centroid = None

    @staticmethod
    def _make_bag_of_centroids(words: List[str], word2centroid: Dict[str, int],
                               bag_of_centroids: Type[np.ndarray]) \
            -> Type[np.ndarray]:
        """
        Converts a list of words into a bag of centroids.

        :param words: A list of words to convert.
        :param word2centroid: A ``word → cluster index`` map.
        :param bag_of_centroids: Output array.
        :return: Bag of centroids representation of ``words``.
        """
        assert isinstance(words, List)
        assert isinstance(word2centroid, Dict)
        assert type(bag_of_centroids) == np.ndarray
        for word in words:
            i = word2centroid.get(word)
            if i is not None:
                bag_of_centroids[i] += 1
            else:
                warnings.warn(('While creating a bag of centroids: {!r} is '
                               'not in the word-index map.').format(word))
        return bag_of_centroids

    # NOTE: This uses the third option from here
    # https://github.com/mlp2018/BagofWords/issues/16. Is this the right
    # way to go?
    def transform(self, reviews: Type[np.ndarray],
                  model: dict) -> Type[np.ndarray]:
        """
        Given a list of reviews, transforms them all to the bag of centroids
        representation.

        :param reviews: Reviews to transform.
        :param model: Word2Vec model to use.
        :return: Array of bag of centroids representations of ``reviews``.
        """
        (num_reviews,) = reviews.shape
        logging.info('Creating bags of centroids...')
        bags = np.zeros((num_reviews, self.kmeans.n_clusters), dtype='float32')
        for (review, bag) in zip(reviews, bags):
            KMeansAverager._make_bag_of_centroids(
                review.split(), self.word2centroid, bag)
        return bags

    def fit_transform(self, reviews: Type[np.ndarray],
                      model: Type[dict]) -> Type[np.ndarray]:
        """
        Given a list of reviews, runs k-means clustering on them and returns
        them in the bag of centroids representation.
        """
        (num_reviews,) = reviews.shape
        num_clusters = int(self.number_clusters_frac * num_reviews)
        self.kmeans = KMeans(n_clusters=num_clusters, **self.kmeans_args)

        vectors = np.array(list(model.values()))

        logging.info('Running k-means + labeling...')
        start = time.time()
        cluster_indices = self.kmeans.fit_predict(vectors)
        end = time.time()
        logging.info('Done with k-means clustering in {:.0f} seconds!'
                     .format(end - start))

        # NOTE: I'm afraid this will go wrong for big word2vec models such as
        # the pre-trained Google's one.
        logging.info('Creating word→index map...')
        self.word2centroid = dict(zip(model.keys(), cluster_indices))

        return self.transform(reviews, model)


# NOTE: @Pauline, you might want to tweak this class to make it work with
# pre-trained Word2Vec models.
class Word2VecVectorizer(object):
    """
    This class implements conversion of reviews to feature vectors using
    Word2Vec model.
    """

    _make_averager_fn = {
        'average': SimpleAverager,
        'k-means': KMeansAverager,
    }

    def __init__(self,
                 averager: str,
                 model_file: str,
                 model_args: Dict[str, Any],
                 averager_args: Dict[str, Any],
                 train_data: Optional[Iterable[str]] = None) -> None:
        assert type(averager) is str
        assert type(model_file) is str
        assert type(model_args) is dict
        assert type(averager_args) is dict
        assert train_data is None or isinstance(train_data, Iterable)
        assert averager in {'average', 'k-means'}

        self.averager = \
            Word2VecVectorizer._make_averager_fn[averager](**averager_args)
        self.model = None
        self.dictionary = None
        # Yes, it's hard-coded. Yes, it's bad, but screw it! :) And what's
        # worse, if you remove the npy extension, everything will break...
        self.dictionary_file = model_file + '.dict.npy'
        # First, of all, retrain the model if we're passed some training data.
        if train_data is not None:
            self.model = Word2VecVectorizer._retrain(
                train_data, model_file, **model_args)
        else:
            # Next step is to check whether we can just load the dictionary...
            if Path(self.dictionary_file).exists():
                self.dictionary = np.load(self.dictionary_file).item()
            else:
                # Well, let's load the model and construct the dictionary
                # afterwards...
                # TODO: This always loads in binary format, because _retrain
                # uses binary format for saving. Not sure if that's OK.
                self.model = KeyedVectors.load_word2vec_format(
                    model_file, binary=True)
        assert self.dictionary is None and self.model is not None \
            or self.model is None and self.dictionary is not None

    @staticmethod
    def _retrain(train_data: Iterable[str], model_file: str, **model_args) \
            -> Word2Vec:
        assert isinstance(train_data, Iterable)
        assert type(model_file) is str
        # Touch the model_file _before_ we start training so that if the path
        # is wrong we don't wait hours for nothing...
        Path(model_file).touch()

        logging.info('Training Word2Vec model...')
        start = time.time()
        model = Word2Vec(reviews2sentences(train_data), **model_args).wv
        stop = time.time()
        logging.info('Done training in {:.0f} seconds!'
                     .format(stop - start))
        logging.info('Saving KeyedVectors model to {!r}...'
                     .format(model_file))
        # TODO: This always saves in binary format. I'm not sure that's what we
        # want.
        #                                                          Tom
        model.save_word2vec_format(model_file, binary=True)
        return model

    @staticmethod
    def _keyed_vectors_to_dict(model: KeyedVectors, reviews: np.ndarray,
                               dictionary_file: Optional[str] = None) \
            -> Dict[str, np.ndarray]:
        """

        """
        assert isinstance(model, KeyedVectors)
        assert type(reviews) is np.ndarray
        assert dictionary_file is None or type(dictionary_file) is str
        words = set(word for review in reviews for word in review.split())
        dictionary = dict(map(lambda x: (x, model[x]),
                              filter(lambda x: x in model, words)))
        if dictionary_file is not None:
            logging.info('Saving word→vector dictionary to {!r}...'
                         .format(dictionary_file))
            np.save(dictionary_file, dictionary)
        return dictionary

    def fit_transform(self, reviews):
        assert self.dictionary is None and self.model is not None \
            or self.model is None and self.dictionary is not None
        # Now that we have reviews at hand, let's make sure we use the
        # dictionary and delete the model
        if self.dictionary is None:
            self.dictionary = Word2VecVectorizer._keyed_vectors_to_dict(
                self.model, reviews, self.dictionary_file)
            self.model = None
        return self.averager.fit_transform(reviews, self.dictionary)

    def transform(self, reviews):
        assert self.dictionary is None and self.model is not None \
            or self.model is None and self.dictionary is not None
        # Same as in fit_transform, let's make sure we use the dictionary and
        # delete the model
        if self.dictionary is None:
            self.dictionary = Word2VecVectorizer._keyed_vectors_to_dict(
                self.model, reviews, self.dictionary_file)
            self.model = None
        return self.averager.transform(reviews, self.dictionary)


def _make_vectorizer(conf):
    vectorizer_str = conf['vectorizer']
    if vectorizer_str == 'bagofwords':
        return CountVectorizer(**conf[vectorizer_str])
    elif vectorizer_str == 'word2vec':
        train_data = \
            _read_data_from(conf['in']['unlabeled'])['review'] \
            if conf['word2vec']['retrain'] else None
        return Word2VecVectorizer(
            conf['word2vec']['strategy'],
            conf['word2vec']['model'],
            train_data=train_data,
            model_args=conf['word2vec'],
            averager_args=conf[conf['word2vec']['strategy']])
    else:
        raise Exception("Unknown vectorizer type.")


def _make_classifier(conf):
    _fn = {
        'logistic-regression':    LogisticRegression,
        'naive-bayes-bagofwords': MultinomialNB,
        'naive-bayes-word2vec':   BernoulliNB,
        'random-forest':          RandomForestClassifier,
        'neural-network':         NeuralNetworkClassifier
    }
    return _fn[conf['classifier']](**conf[conf['classifier']])


class NeuralNetworkClassifier(object):

    def __init__(self, batch_size, n_steps, n_hidden_units1, n_hidden_units2,
                 n_classes):
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_hidden_units1 = n_hidden_units1
        self.n_hidden_units2 = n_hidden_units2
        self.n_classes = n_classes
        self.model_description = str(n_hidden_units1) + '_' + str(n_hidden_units2)
        self.model_dir = str(_PROJECT_ROOT / 'models' / 'nn' / self.model_description)

    def set_up_architecture(self, train_data_features, train_sentiments):

        # Convert the scarce scipy feature matrices to pandas dataframes
        print(train_data_features.shape)
        train_df = pd.DataFrame(train_data_features.toarray())

        # Convert column names from numbers to strings
        train_df.columns = train_df.columns.astype(str)

        # Create feature columns which describe how to use the input
        feat_cols = []
        for key in train_df.keys():
            feat_cols.append(tf.feature_column.numeric_column(key=key))

        # Set up classifier with two hidden unit layers
        classifier = tf.estimator.DNNClassifier(
                                        feature_columns=feat_cols,
                                        hidden_units=[self.n_hidden_units1,
                                                      self.n_hidden_units2],
                                        n_classes=self.n_classes,
                                        model_dir=self.model_dir)

        return train_df, classifier

    def check_saves(self):
        pass

    def shape_train_input(self, features, labels, batch_size):
        """An input function for training"""

        # Convert the input to a dataset
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle, repeat, and batch the examples
        dataset = dataset.shuffle(1000).repeat().batch(self.batch_size)

        return dataset

    def fit(self, train_data_features, train_sentiments):

        train_df, new_classifier = self.set_up_architecture(train_data_features,
                                                   train_sentiments)

        self.classifier = new_classifier.train(input_fn=
                                      lambda:self.shape_train_input(train_df,
                                                              train_sentiments,
                                                              self.batch_size),
                                                              steps=self.n_steps)

        return self

    def shape_pred_input(self, features, batch_size):
        """An input function for evaluation or prediction"""

        features=dict(features)

        # Convert the inputs to a dataset
        dataset = tf.data.Dataset.from_tensor_slices(features)

        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(self.batch_size)

        print(dataset)

        return dataset

    def predict(self, test_data_features):

        # Convert the scarce scipy feature matrices to pandas dataframes
        test_df = pd.DataFrame(test_data_features.toarray())

        # Convert column names from numbers to strings
        test_df.columns = test_df.columns.astype(str)

        predictions = self.classifier.predict(input_fn=
                                    lambda:self.shape_pred_input(test_df,
                                                            self.batch_size))

        predicted_labels = []

        for pred_dict in predictions:
            predicted_labels.append(pred_dict['class_ids'][0])

        return predicted_labels


# def set_of_words(reviews):
#     '''
#     create the set of words
#     :param reviews:
#
#     '''
#     split_reviews = []
#     for review in reviews:
#         split_reviews.append(review.split())
#
#     flat_word_list = [item for sublist in split_reviews for item in sublist]
#     unique_words = set(flat_word_list)
#
#     return unique_words

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    if conf['run']['type'] == 'optimization':
        train_data = _read_data_from(conf['in']['labeled'])
        ids = np.array(train_data['id'], dtype=np.unicode_)
        reviews = clean_up_reviews(train_data['review'],
                                   conf['run']['remove_stopwords'],
                                   conf['in']['clean'])
        sentiments = np.array(train_data['sentiment'], dtype=np.bool_)

        def mk_vectorizer():
            return _make_vectorizer(conf)

        def mk_classifier():
            return _make_classifier(conf)

        optimization_run(reviews, sentiments,
                         mk_vectorizer, mk_classifier,
                         conf['run'],
                         conf['out']['wrong_result'])
    elif conf['run']['type'] == 'submission':
        train_data = _read_data_from(conf['in']['labeled'])
        test_data = _read_data_from(conf['in']['test'])
        # [:100] # If we don't slice until :100, a ValueError is raised.
        # Why does the test data need to have the same size as the training
        # data?
        ids = np.array(test_data['id'], dtype=np.unicode_)
        reviews = clean_up_reviews(train_data['review'],
                                   conf['run']['remove_stopwords'],
                                   conf['in']['clean'])
        sentiments = np.array(train_data['sentiment'], dtype=np.bool_)
        test_reviews = clean_up_reviews(test_data['review'],
                                        conf['run']['remove_stopwords'],
                                        conf['in']['clean'])

        def mk_vectorizer():
            return _make_vectorizer(conf)

        def mk_classifier():
            return _make_classifier(conf)

        submission_run(reviews, sentiments,
                       test_reviews,
                       ids,
                       mk_vectorizer, mk_classifier,
                       conf['out']['result'])
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    cProfile.run('main()')
