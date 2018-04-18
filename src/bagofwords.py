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
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam, SGD
import nltk.corpus
import nltk.tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
# from scipy.sparse import csc_matrix
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score

# import cProfile


def _get_sklearn_version() -> Tuple[int, int, int]:
    """
    Returns the version of scikit-learn as a tuple.
    """
    (release, major, minor) = sklearn.__version__.split('.')
    return int(release), int(major), int(minor)


if _get_sklearn_version() >= (0, 19, 0):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import StratifiedShuffleSplit
else:  # We have an old version of sklearn...
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.cross_validation import StratifiedShuffleSplit


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
        'type':             'submission',
        # How many splits to use in the StratifiedKFold
        'number_splits':    3,
        # When preprocessing the reviews, should we remove the stopwords?
        'remove_stopwords': True,
        # Should we cache the preprocessed reviews?
        'cache_clean':      True,
        # After the running the StratifiedKFold on the 90%, should we test the
        # result on the remaining 10?
        'test_10':          False,
        # Random seed used for the 90-10 split.
        'random':           54,
        # How many percent of the data should be left out for testing? I.e.
        # what is "10" in the 90-10 split.
        'alpha':            0.1,
    },
    # Type of the vectorizer, one of {'word2vec', 'bagofwords'}
    'vectorizer': 'word2vec',
    # Type of the classifier to use, one of
    # {'random-forest', 'logistic-regression', 'feed-forward', 'convolutional'}
    'classifier': 'convolutional',
    # Options specific to the bagofwords vectorizer.
    'bagofwords': {},
    # Options specific to the word2vec vectorizer.
    'word2vec': {
        # File name where to save/read the model to/from.
        'model_file': str(_PROJECT_ROOT / 'data'
                          / 'GoogleNews-vectors-negative300.bin'),
        # File name where to save/read the "compact" version to/from.
        'cache_file': str(_PROJECT_ROOT / 'data'
                          / 'GoogleNews-vectors-negative300.compact.bin'),
        # Retrain the model every time?
        'retrain':    False,
        # Averaging strategy to use, one of {'average', 'k-means', 'dummy'}
        # NOTE: Random Forest, Logistic Regression, and Feed Forward NN work
        # with 'average' and 'k-means'; Convolutional NN works __only__ with
        # 'dummy'.
        'strategy':   'dummy',
    },
    # Options specific to the random forest classifier.
    'random-forest': {
        'n_estimators': 1000,
        'n_jobs':       (-1),
        'max_depth':    10,
        'max_features': 'log2',
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
    # Options specific to Andre's convolutional NN classifier.
    'convolutional': {
        'n_units':      250,
        'n_filters':    90,
        'n_epochs':     4,
        'verbose':      2,
        'kernel_size':  6,
        'batch_size':   128,
        'use_bias':     True,
        'loss':         'binary_crossentropy',
        'optimizer':    Adam(lr=0.0021),
        'metrics':      ['accuracy'],
    },
    # Options specific to feed forward neural network classifier.
    'feed-forward': {
        'n_hidden_units1': 16,
        'n_hidden_units2': 10,
        'batch_size':      32,
        'n_epochs':        200,
        'loss':            'binary_crossentropy',
        'optimizer':       SGD(lr=0.01),
        'metrics':         ['accuracy'],
    },
    # Dummy entry for dummy averager :)
    'dummy': {},
    # Options specific to the "average" averaging strategy.
    'average': {},
    # Options specific to the "k-means" averaging strategy.
    'k-means': {
        'number_clusters_frac': 0.2,  # NOTE: This argument is required!
        'warn_on_missing': False,
        # 'max_iter':             100,
        # 'n_jobs':               4,
    },
}


# If you have a local conf.py which defines the configuration options, it will
# be used. Otherwise, this falls back to defaults.
# NOTE: See _DEFAULT_CONFIG for the format of the configuratio options.
try:
    from conf import conf
    print("Imported local config file.", file=sys.stderr)
except ImportError:
    conf = _DEFAULT_CONFIG
    print("Used the default config.", file=sys.stderr)


# TODO: Fix this.
# Turn off warnings about Beautiful Soup (Johanna has checked all of them
# manually)
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


def _read_data_from(path: str) -> pd.DataFrame:
    logging.info('Reading data from {!r}...'.format(path))
    return pd.read_csv(path, header=0, delimiter='\t', quoting=3)


class LemmatizationWithPOSTagger(object):
    def __init__(self):
        self.lemmatizer = nltk.WordNetLemmatizer()

    def get_wordnet_pos(self, treebank_tag):
        """
        Convert Part of Speech Tags to comply with the expectations of wordnet
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN

    def pos_tag(self, tokens):
        """
        Method to obtain POS-Tags

        """
        # find the pos tagginf for each tokens [('What', 'WP'), ('can', 'MD'), ('I', 'PRP') ....
        pos_tokens = [nltk.pos_tag(token) for token in tokens]

        # TODO: The following looks ugly...
        # lemmatization using pos tagg
        # convert into feature set of [('What', 'What', ['WP']), ('can', 'can', ['MD']), ... ie [original WORD, Lemmatized word, POS tag]
        pos_tokens = [ [(word, self.lemmatizer.lemmatize(word,self.get_wordnet_pos(pos_tag)), [pos_tag]) for (word,pos_tag) in pos] for pos in pos_tokens]
        return pos_tokens


class ReviewPreprocessor(object):
    """
    :py:class:`ReviewPreprocessor` is an utility class for processing raw HTML
    text into segments for further learning.
    """

    _stopwords = set(nltk.corpus.stopwords.words('english'))
    _tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    @staticmethod
    def _striphtml(text: str) -> str:
        text = re.sub(r'(www.|http[s]?:\/)(?:[a-zA-Z]|[0-9]|[$-_@.&+]'
                      r'|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                      r'', text)
        text = BeautifulSoup(text, 'html.parser').get_text()
        return text

    @staticmethod
    def _2wordlist(text: str, remove_stopwords: bool = False) -> List[str]:
        text = re.sub('[^a-zA-Z]', ' ', text)
        words = text.lower().split()
        if remove_stopwords:
            words = [w for w in words if w not in
                     ReviewPreprocessor._stopwords]
        # TODO: This is ugly... Is it even correct?
        words = [words]
        lemma_pos_token = LemmatizationWithPOSTagger().pos_tag(words)
        words = [[x[1] for x in el] for el in lemma_pos_token]
        words = words[0]
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


def submission_run(reviews: np.ndarray,
                   sentiments: np.ndarray,
                   test_reviews: np.ndarray,
                   ids: np.ndarray,
                   mk_vectorizer: Callable[[], Any],
                   mk_classifier: Callable[[], Any],
                   prediction_file: str) -> np.ndarray:
    """
    :param reviews:         Array of raw reviews texts.
    :type reviews:          'numpy.ndarray' of shape ``(N,)`` of type ``str``
    :param sentiments:      Array of review sentiments.
    :type sentiments:       'numpy.ndarray' of shape ``(N,)`` of type ``bool``
    :param test_reviews:    Array of test review texts.
    :type test_reviews:     'numpy.ndarray' of shape ``(N,)`` of type ``str``
    :param ids:             Array of review identifiers.
    :type ids:              'numpy.ndarray' of shape ``(N,)`` of type ``str``
    :param mk_vectorizer:   Factory function to create a new vectorizer.
    :type mk_vectorizer:    Callable[[], Vectorizer]
    :param mk_classifier:   Factory function to create a new classifier.
    :type mk_classifier:    Callable[[], Classifier]
    :param prediction_file: File where the predicted sentiments are saved to.
    :type prediction_file:  str
    """
    _, prediction = run_one_fold(
        (reviews, sentiments), test_reviews, mk_vectorizer(),
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


def run_one_fold(train_data: Tuple[np.ndarray, np.ndarray],
                 test_data: Tuple[np.ndarray, np.ndarray],
                 vectorizer: Any, classifier: Any) \
        -> Tuple[float, np.ndarray]:
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


def split_90_10(data: Tuple[np.ndarray, np.ndarray],
                alpha: float, seed: int = 42) \
        -> Tuple[Tuple[np.ndarray, np.ndarray],
                 Tuple[np.ndarray, np.ndarray]]:
    """
    Despite the very descriptive name this function does the ``1-alpha`` -
    ``alpha`` split rather than the ``90%`` - ``10%`` one.

    :param data:  The data to split.
    :param alpha: Percentage of data to use for testing.
    :param seed:  Random seed to use for splitting.
    :return: ``((reviews to train on, sentiments to train on),
                (reviews to test on, sentiments to test on))``.
    """
    assert alpha > 0 and alpha < 1
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


def run_a_couple_of_folds(data: Tuple[np.ndarray, np.ndarray],
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


def optimization_run(reviews: np.ndarray,
                     sentiments: np.ndarray,
                     mk_vectorizer: Callable[[], Any],
                     mk_classifier: Callable[[], Any],
                     conf: Dict[str, Any],
                     wrong_prediction_file: str) -> np.ndarray:
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
    _, _, wrong_idx = run_a_couple_of_folds(
        data_90, mk_vectorizer, mk_classifier, conf['number_splits'])
    wrong_reviews = reviews_90[wrong_idx]
    if conf['test_10']:
        logging.info('Training on 90% and testing on 10%...')
        _, prediction_10 = run_one_fold(
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
    def _make_avg_feature_vector(words: str,
                                 known_words: KeyedVectors,
                                 average_vector: Optional[np.ndarray] = None) \
            -> np.ndarray:
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
        word_count = sum(
            1 for _ in map(lambda x: np.add(average_vector, known_words[x],
                                            out=average_vector),
                           filter(lambda x: x in known_words, words.split()))
        )
        return np.divide(average_vector, float(word_count), out=average_vector)

    def transform(self, reviews: np.ndarray, model: KeyedVectors) \
            -> np.ndarray:
        # TODO: Docs
        (_, number_features) = model.syn0.shape
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

    def fit_transform(self, reviews: np.ndarray, model: KeyedVectors)\
            -> np.ndarray:
        """
        :py:class:`SimpleAverager` has no state, and :py:func:`fit_transform`
        thus simply calls :py:func:`transform`.
        """
        return self.transform(reviews, model)


class KMeansAverager(object):
    """
    This class implements the 'k-means' averaging strategy for reducing a list
    of word2vec vectors to a single one.
    """

    def __init__(self, number_clusters_frac: float,
                 warn_on_missing: bool, **kwargs: dict) -> None:
        self.number_clusters_frac = number_clusters_frac
        self.warn_on_missing = warn_on_missing
        self.kmeans_args = kwargs
        self.kmeans = None
        self.word2centroid = None

    @staticmethod
    def _make_bag_of_centroids(words: Iterable[str],
                               word2centroid: Dict[str, int],
                               bag_of_centroids: np.ndarray,
                               warn_on_missing: bool = False) \
            -> np.ndarray:
        """
        Given a stream of words and a mapping of words to centroid indices,
        converts these words to "bag of centroids" representation.

        :param words:            A list of words to convert.
        :type words:             Iterable[str]
        :param word2centroid:    A ``word → cluster index`` map.
        :type word2centroid:     Dict[str, int]
        :param bag_of_centroids: Output array.
        :type bag_of_centroids:  np.ndarray
        :param warn_on_missing:  Whether to issue warnings is some words are
                                 not in the `word2centroid` dictionary.
        :type warn_on_missing:   bool
        :return:                 Bag of centroids representation of ``words``.
        :rtype:                  np.ndarray
        """
        for word in words:
            i = word2centroid.get(word)
            if i is not None:
                bag_of_centroids[i] += 1
            elif warn_on_missing:
                warnings.warn(('While creating a bag of centroids: {!r} is '
                               'not in the word→index map.').format(word))
        return bag_of_centroids

    # NOTE: This uses the third option from here
    # https://github.com/mlp2018/BagofWords/issues/16. Is this the right
    # way to go?
    def transform(self, reviews: np.ndarray,
                  model: KeyedVectors) -> np.ndarray:
        """
        Given a list of reviews and a word2vec mapping, transforms all reviews
        to the bag of centroids representation.

        :param reviews: Reviews to transform.
        :type reviews:  np.ndarray
        :param model:   ``word→vector`` mapping.
        :type model:    Dict[str, np.ndarray]
        :return:        Array of reviews in the "bag of centroids"
                        representation.
        """
        (num_reviews,) = reviews.shape
        logging.info('Creating bags of centroids...')
        bags = np.zeros((num_reviews, self.kmeans.n_clusters), dtype='float32')
        for (review, bag) in zip(reviews, bags):
            KMeansAverager._make_bag_of_centroids(
                review.split(), self.word2centroid, bag, self.warn_on_missing)
        return bags

    def fit_transform(self, reviews: np.ndarray,
                      model: KeyedVectors) -> np.ndarray:
        """
        Given a list of reviews and a word2vec mapping, runs k-means clustering
        on the vector representation of reviews and converts them to the bag of
        centroids representation.

        :param reviews: Reviews to transform.
        :type reviews:  np.ndarray
        :param model:   ``word→vector`` mapping.
        :type model:    Dict[str, np.ndarray]
        :return:        Array of reviews in the "bag of centroids"
                        representation.
        """
        (num_reviews,) = reviews.shape
        num_clusters = int(self.number_clusters_frac * num_reviews)
        self.kmeans = MiniBatchKMeans(
            n_clusters=num_clusters, **self.kmeans_args)
        vectors = model.syn0
        logging.info('Running k-means + labeling...')
        start = time.time()
        cluster_indices = self.kmeans.fit_predict(vectors)
        end = time.time()
        logging.info('Done with k-means clustering in {:.0f} seconds!'
                     .format(end - start))
        logging.info('Creating word→index map...')
        self.word2centroid = dict(zip(model.index2word, cluster_indices))
        return self.transform(reviews, model)


class DummyAverager(object):
    """
    Averager that does not average :)
    """

    def __init__(self):
        pass

    def transform(self, reviews, model):
        return reviews, model

    def fit_transform(self, reviews, model):
        return reviews, model


class Word2VecVectorizer(object):
    """
    This class implements conversion of reviews to feature vectors using
    Word2Vec model.
    """

    _make_averager_fn = {
        'average': SimpleAverager,
        'k-means': KMeansAverager,
        'dummy':   DummyAverager,
    }

    def __init__(self,
                 strategy: str,
                 model_file: str,
                 model_args: Optional[Dict[str, Any]] = None,
                 averager_args: Optional[Dict[str, Any]] = None,
                 cache_file: Optional[str] = None,
                 retrain: bool = False,
                 train_data: Optional[Iterable[str]] = None) -> None:
        self.averager = \
            Word2VecVectorizer._make_averager_fn[strategy](**averager_args)
        self.model = None
        self.cache_file = cache_file
        self.is_compact = False
        # First of all, retrain the model if we're passed some training data.
        if train_data is not None:
            self.model = Word2VecVectorizer._retrain(
                train_data, model_file, **model_args)
        else:
            # If we already have our model cached, let's just load it.
            if cache_file is not None and Path(cache_file).exists():
                self.model = KeyedVectors.load_word2vec_format(
                    cache_file, binary=True)
                self.is_compact = True
            else:
                # Well, let's load the model and construct remove all
                # unnecessary stuff afterwards...
                # TODO: This always loads in binary format, because _retrain
                # uses binary format for saving. Not sure if that's OK.
                self.model = KeyedVectors.load_word2vec_format(
                    model_file, binary=True)

    @staticmethod
    def _retrain(train_data: Iterable[str], model_file: str, **model_args) \
            -> Word2Vec:
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
    def _compress_keyed_vectors(model: KeyedVectors, reviews: np.ndarray,
                                cache_file: Optional[str] = None) -> None:
        required_words = set(word for review in reviews
                             for word in review.split())
        for (i, word) in enumerate(model.index2word):
            assert model.vocab[word].index == i

        # This is a really shady piece :)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # First of all, extract all the words we really need
        index2word = [w for w in model.index2word if w in required_words]

        def update_index(item, index):
            item.index = index
            return item
        # In the first approximation, we want
        # vocab = dict((w, model.vocab[w]) for w in index2word)
        # i.e. to keep the data for words which are in index2word. The problem
        # is that Vocab type has a field index which stores word's index in the
        # syn0 vector and index2word list. We, obbiously, break this indexing
        # by rebuilding index2word from scratch. So here we manually
        # reconstruct correct indexing.
        vocab = dict((w, update_index(model.vocab[w], i))
                     for (i, w) in enumerate(index2word))
        # Now, extract correct indices as an np.ndarray
        indices = np.fromiter(
            map(lambda x: x.index, vocab.values()), dtype=int)
        # Only keep feature vectors for the words we really need.
        vectors = model.syn0[indices]
        # Do the actual update
        model.index2word = index2word
        model.vocab = vocab
        model.syn0 = vectors
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # end shady code...

        if cache_file is not None:
            model.save_word2vec_format(cache_file, binary=True)
        return model

    def fit_transform(self, reviews):
        if not self.is_compact:
            self.model = Word2VecVectorizer._compress_keyed_vectors(
                self.model, reviews, self.cache_file)
            self.is_compact = True
        return self.averager.fit_transform(reviews, self.model)

    def transform(self, reviews):
        if not self.is_compact:
            self.model = Word2VecVectorizer._compress_keyed_vectors(
                self.model, reviews, self.cache_file)
            self.is_compact = True
        return self.averager.transform(reviews, self.model)


def _make_vectorizer(conf):
    vectorizer_str = conf['vectorizer']
    if vectorizer_str == 'bagofwords':
        return CountVectorizer(**conf[vectorizer_str])
    elif vectorizer_str == 'word2vec':
        train_data = \
            _read_data_from(conf['in']['unlabeled'])['review'] \
            if conf['word2vec']['retrain'] else None
        return Word2VecVectorizer(
            **conf['word2vec'],
            averager_args=conf[conf['word2vec']['strategy']],
            train_data=train_data)
    else:
        raise Exception("Unknown vectorizer type.")


def _make_classifier(conf):
    _fn = {
        'logistic-regression':    LogisticRegression,
        'naive-bayes-bagofwords': MultinomialNB,
        'naive-bayes-word2vec':   BernoulliNB,
        'random-forest':          RandomForestClassifier,
        'convolutional':          ConvNNClassifier,
        'feed-forward':           FeedForwardNNClassifier,
    }
    return _fn[conf['classifier']](**conf[conf['classifier']])


class ConvNNClassifier(object):
    def __init__(self, n_units, n_filters, kernel_size,
                 use_bias, n_epochs, batch_size, loss,
                 optimizer, metrics, verbose):
        self.n_units = n_units
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.callbacks = [EarlyStopping(patience=1)]
        self.max_words = 500
        self.model = None
        self.history = None

    def _construct(self, word2vec):
        (n_words, n_features) = word2vec.vectors.shape
        model = Sequential()
        layer = Embedding(input_dim=n_words,
                          output_dim=n_features,
                          input_length=self.max_words,
                          weights=[word2vec.vectors],
                          trainable=False)
        model.add(layer)
        model.add(Conv1D(filters=self.n_filters,
                         kernel_size=self.kernel_size,
                         padding='same',
                         activation='relu',
                         use_bias=self.use_bias))
        model.add(MaxPooling1D(pool_size=self.max_words, strides=2))
        model.add(Flatten())
        model.add(Dense(self.n_units,
                        activation='relu',
                        use_bias=self.use_bias))
        model.add(Dense(1, activation='sigmoid', use_bias=self.use_bias))
        model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        logging.info(model.summary())
        return model

    @staticmethod
    def _review_to_tensor(review, word2vec, max_words):
        tensor = np.full((max_words,), fill_value=max_words)  # ?????
        for (i, word) in enumerate(filter(lambda x: x in word2vec,
                                          review.split())):
            # TODO: The following `if` is ungly, fix it please!
            if i >= max_words:
                break
            if word in word2vec:
                tensor[i] = word2vec.vocab[word].index
        return tensor

    def fit(self, train_data, labels):
        reviews, word2vec = train_data
        logging.info('Converting reviews to tensors...')
        reviews = np.array(list(map(
            lambda x: ConvNNClassifier._review_to_tensor(x, word2vec,
                                                         self.max_words),
            reviews)))
        logging.info('Done!')
        self.model = self._construct(word2vec)
        self.history = self.model.fit(x=reviews, y=labels,
                                      validation_data=None,
                                      epochs=self.n_epochs,
                                      batch_size=self.batch_size,
                                      verbose=self.verbose,
                                      callbacks=self.callbacks)
        return self

    def predict(self, train_data):
        reviews, word2vec = train_data
        (n_reviews,) = reviews.shape
        logging.info('Converting reviews to tensors...')
        reviews = np.array(list(map(
            lambda x: ConvNNClassifier._review_to_tensor(x, word2vec,
                                                         self.max_words),
            reviews)))
        logging.info('Done!')
        return self.model.predict(reviews).reshape((n_reviews,))


class FeedForwardNNClassifier(object):

    def __init__(self, n_hidden_units1, n_hidden_units2, batch_size,
                 n_epochs, optimizer, loss, metrics):
        self.n_hidden_units1 = n_hidden_units1
        self.n_hidden_units2 = n_hidden_units2
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.model = None
        self.history = None

    def _construct(self, n_features):
        model = Sequential()
        model.add(Dense(self.n_hidden_units1,
                        input_dim=n_features,
                        activation='relu',
                        kernel_initializer='uniform'))
        model.add(Dense(self.n_hidden_units2,
                        activation='relu',
                        kernel_initializer='uniform'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss=self.loss, optimizer=self.optimizer,
                      metrics=self.metrics)
        logging.info(model.summary())
        return model

    def fit(self, reviews, sentiments):
        (_, n_features) = reviews.shape
        self.model = self._construct(n_features)
        self.history = self.model.fit(x=reviews, y=sentiments,
                                      epochs=self.n_epochs,
                                      batch_size=self.batch_size)
        return self

    def predict(self, reviews):
        (n_reviews, _) = reviews.shape
        return self.model.predict(reviews).reshape((n_reviews,))


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    if conf['run']['type'] == 'optimization':
        train_data = _read_data_from(conf['in']['labeled'])
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
        ids = np.array(test_data['id'], dtype=np.unicode_)
        reviews = clean_up_reviews(train_data['review'],
                                   conf['run']['remove_stopwords'],
                                   conf['in']['clean'])
        sentiments = np.array(train_data['sentiment'], dtype=np.bool_)
        test_reviews = clean_up_reviews(test_data['review'],
                                        conf['run']['remove_stopwords'])

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
    # cProfile.run('main()')
    main()
