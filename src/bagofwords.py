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
from typing import Type, Any, Iterable, Tuple, List, Set, Dict, Callable
import warnings

from bs4 import BeautifulSoup
from gensim.models import Word2Vec, KeyedVectors
import nltk.corpus
import nltk.tokenize
import numpy as np
import pandas as pd
# from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_extraction.text import VectorizerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score

if sys.version_info >= (3, 6):
    from sklearn.model_selection import StratifiedKFold
else:  # We have an old version of sklearn...
    from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit


def _get_current_file_dir() -> Path:
    """Returns the directory of the script."""
    try:
        return Path(os.path.realpath(__file__)).parent
    except(NameError):
        return Path(os.getcwd())


# Project root directory, i.e. the github repo directory.
_PROJECT_ROOT = _get_current_file_dir() / '..'


# Default configuration options.
_DEFAULT_CONFIG = {
    'in': {
        'labeled':   str(_PROJECT_ROOT / 'data' / 'labeledTrainData.tsv'),
                   # str(_PROJECT_ROOT / 'data' / 'small_train_data_set.tsv'),
        'unlabeled': str(_PROJECT_ROOT / 'data' / 'unlabeledTrainData.tsv'),
        'test':      str(_PROJECT_ROOT / 'data' / 'testData.tsv'),
        'clean':     str(_PROJECT_ROOT / 'data' / 'cleanReviews.tsv'),
    },
    'out': {
        'result':       str(_PROJECT_ROOT / 'results' / 'Prediction.csv'),
        'wrong_result': str(_PROJECT_ROOT / 'results' / 'FalsePrediction.csv'),
    },
    'vectorizer': {
        # Type of the vectorizer, one of {'word2vec', 'bagofwords'}
        'type': 'word2vec',
        'args': {},
        # 'args': {
        #     'size':      300,
        #     'min_count': 40,
        #     'window':    10,
        #     'sample':    1.E-3,
        #     'workers':   4,
        #     'seed':      1,
        # },
    },
    # 'vectorizer': {
    #     'type': 'bagofwords'
    #     'args': {
    #         'analyzer': 'word',
    #         'tokenizer': None,
    #         'stop_words': None,
    #         'max_features': 5000
    #     },
    # }
    'classifier': {
        # Type of the classifier to use, one of {'random-forest'}
        # NOTE: Currently, 'random-forest' is the only working option.
        'type': 'random-forest',
        'args': {
            'n_estimators': 100,
            # 'max_features': 20000,
            'n_jobs':       4,
        },
    },
    'run': {
        # Type of the run, one of {'optimization', 'submission'}
        # NOTE: Currently, only optimization run is implemented.
        'type': 'submission',
        'number_splits': 3,
        'remove_stopwords': False,
        'cache_clean': True,
        'test_10': True,
        'random': 42,
        'alpha': 0.1,
    },
    'bagofwords': {},
    'word2vec': {
        'model':    str(_PROJECT_ROOT / 'results'
                                      / '300features_40minwords_10context'),
        'retrain':  False,
        # Averaging strategy to use, one of {'average', 'k-means', 'mini-batch-k-means'}
        'strategy': 'mini-batch-k-means'
    },
    'k-means': {
        'number_clusters_frac': 0.2,  # NOTE: This argument is required!
        'max_iter':             100,
        'n_jobs':               2,
    },
    'mini-batch-k-means': {
        'number_clusters_frac': 0.2,
        'init_size' : 300, 
        'batch_size': 100,
            
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
    ReviewPreprocessor is an utility class for processing raw HTML text into
    segments for further learning.
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
        return words

    @staticmethod
    def review2wordlist(review: str, remove_stopwords: bool = False) \
            -> List[str]:
        """
        Given a review, parses it as HTML, removes all URLs and
        non-alphabetical characters, and optionally removes the stopwords. The
        review is then split into lowercase words and the resulting list is
        returned.

        :param str review:            The review as raw HTML
        :param bool remove_stopwords: Whether to remove stopwords.
        :return:                      Review split into words.
        :rtype:                       ``List[str]``
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
        Given a review, splits the review into sentences, where each sentence
        is a list of words.

        :param str review:            The review as raw HTML.
        :param bool remove_stopwords: Whether to remove stopwords.
        :return:                      Review split into sentences.
        :rtype:                       ``Iterable[List[str]]``.
        """
        assert type(review) == str
        assert type(remove_stopwords) == bool

        # TODO: First preprocess using BeautifulSoup!

        raw_sentences = ReviewPreprocessor._tokenizer.tokenize(
            ReviewPreprocessor._striphtml(review))
        return map(
            lambda x: ReviewPreprocessor._2wordlist(x, remove_stopwords),
            filter(lambda x: x, raw_sentences))


# TODO: This function should probably also do the stemming, see
# https://github.com/mlp2018/BagofWords/issues/7.
def clean_up_reviews(reviews: Iterable[str],
                     remove_stopwords: bool = True,
                     compute_only: bool = False) -> Type[np.ndarray]:
    """
    Given an list of reviews, either loads pre-computed reviews
    or applies
    :py:func:`ReviewPreprocessor.review2wordlist` to each review and
    concatenates it back into a single string.
    and saves to file 'cleaned_reviews'

    :param reviews:               Reviews to clean up.
    :type reviews:                Iterable[str]
    :param bool remove_stopwords: Whether to remove the stopwords.
    :param bool compute_only:     Whether to save result and load from file
                                  if exist.
    :return:                      Iterable of clean reviews.
    :rtype:                       Iterable[str]
    """
    assert isinstance(reviews, Iterable)
    assert type(remove_stopwords) == bool
    assert type(compute_only) == bool
    clean_file = conf['in']['clean']
    if not compute_only and Path(clean_file).exists():
        return _read_data_from(clean_file)['review'].values
    logging.info('Cleaning and parsing the reviews...')
    review = np.array(list(map(
        lambda x: ' '.join(
            ReviewPreprocessor.review2wordlist(x, remove_stopwords)),
        reviews)))
    if not compute_only:
        logging.info('Saving clean data to file "cleanReviews.tsv" ...')
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
    assert type(remove_stopwords) == bool

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

    :param data: The data to split.
    :param alpha: Percentage of data to use for testing.
    :param seed: Random seed to use for splitting.
    :return: ``((reviews to train on, sentiments to train on),
                (reviews to test on, sentiments to test on))``.
    """
    assert 0 < alpha and alpha < 1
    reviews, labels = data
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=alpha, random_state=seed)
    for train_index, test_index in sss.split(reviews, labels):
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

    if sys.version_info >= (3, 6):
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
    This class implements the 'average' strategy for reducing a list of
    word2vec vectors into a single one.
    """

    def __init__(self):
        pass

    @staticmethod
    def _make_avg_feature_vector(
            words: List[str], model: Type[KeyedVectors], known_words: Set[str],
            average_vector: Type[np.ndarray]) -> Type[np.ndarray]:
        """
        Given a list of words, returns the their average.

        :param words: Words to average.
        :type words: List[str]
        :param model: Words representation, i.e. the "word vectors"-part of
                      Word2Vec model.
        :type model: KeyedVectors,
        :param known_words: Set of all words that the model knows.
        :type known_words: Set[str]
        :param average_vector: Pre-allocated zero-initialised vector. It will
                               contain the average of ``words``.
        :type average_vector: np.ndarray
        :return: The average of ``words``.
        :rtype: np.ndarray
        """
        assert isinstance(words, List)
        assert type(model) == KeyedVectors
        assert isinstance(known_words, Set)
        assert type(average_vector) == np.ndarray
        word_count = sum(
            1 for _ in map(lambda x: np.add(average_vector, model[x],
                                            out=average_vector),
                           filter(lambda x: x in known_words, words))
        )
        return np.divide(average_vector, float(word_count), out=average_vector)

    def transform(self, reviews: Type[np.ndarray],
                  model: Type[KeyedVectors]) -> Type[np.ndarray]:
        """
        Given a list of reviews and a word2vec model, returns an array of
        average feature vectors.

        :param reviews: Reviews to transform.
        :type reviews: 'numpy.ndarray' of ``list`` of shape ``(#reviews,)``
        :param model: Word2Vec model.
        :type model: KeyedVectors
        :return: Array of average feature vectors.
        :rtype: 'numpy.ndarray' of shape ``(#reviews, #features)``
        """
        assert type(reviews) == np.ndarray
        assert type(model) == KeyedVectors
        (_, number_features) = model.syn0.shape
        (number_reviews,) = reviews.shape
        # NOTE: Will this work OK for large models?
        known_words = set(model.index2word)

        feature_vectors = np.zeros(
            (number_reviews, number_features), dtype='float32')
        for (i, (review, vector)) in enumerate(zip(reviews, feature_vectors)):
            if i % 1000 == 0:
                logging.info('PROGRESS: At review #{} of {}...'
                             .format(i, number_reviews))
            SimpleAverager._make_avg_feature_vector(
                review, model, known_words, vector)
        return feature_vectors

    def fit_transform(self, reviews: np.ndarray, model: KeyedVectors):
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

    def __init__(self, **kwargs):
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
                  model: KeyedVectors) -> Type[np.ndarray]:
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
                      model: Type[KeyedVectors]) -> Type[np.ndarray]:
        """
        Given a list of reviews, runs k-means clustering on them and returns
        them in the bag of centroids representation.
        """
        (num_reviews,) = reviews.shape
        num_clusters = int(self.number_clusters_frac * num_reviews)
        if conf['word2vec']['strategy'] == 'mini-batch-k-means':
            self.kmeans = MiniBatchKMeans(n_clusters=num_clusters, )
        if conf['word2vec']['strategy'] == 'k-means':
            self.kmeans = KMeans(n_clusters=num_clusters, **self.kmeans_args)

        logging.info('Running k-means + labeling...')
        start = time.time()
        cluster_indices = self.kmeans.fit_predict(model.syn0)
        end = time.time()
        logging.info('Done with k-means clustering in {:.0f} seconds!'
                     .format(end - start))

        # NOTE: I'm afraid this will go wrong for big word2vec models such as
        # the pre-trained Google's one.
        logging.info('Creating word→index map...')
        self.word2centroid = dict(zip(model.index2word, cluster_indices))

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
        'mini-batch-k-means': KMeansAverager,
    }

    def __init__(self, averager: str, model_file: str,
                 train_data: Iterable[str] = None,
                 model_args={}, averager_args={}):
        assert averager in {'average', 'k-means', 'mini-batch-k-means'}
        self.model = None
        self.averager = None

        if train_data is not None:
            logging.info('Training Word2Vec model...')
            start = time.time()
            self.model = Word2Vec(reviews2sentences(train_data), **model_args)
            stop = time.time()
            logging.info('Done training in {:.0f} seconds!'
                         .format(stop - start))
            if model_file is not None:
                logging.info('Saving Word2Vec model to {!r}...'
                             .format(model_file))
                self.model.save(model_file)
        else:
            # TODO: We do not really need the whole Word2Vec model,
            # KeyedVectors should suffice.
            self.model = Word2Vec.load(model_file, **model_args)

        self.model = self.model.wv
        self.averager = \
            Word2VecVectorizer._make_averager_fn[averager](**averager_args)

    def fit_transform(self, reviews):
        return self.averager.fit_transform(reviews, self.model)

    def transform(self, reviews):
        return self.averager.transform(reviews, self.model)


def _make_vectorizer(conf):
    type_str = conf['vectorizer']['type']
    args = conf['vectorizer']['args']
    if type_str == 'bagofwords':
        return CountVectorizer(**args)
    elif type_str == 'word2vec':
        train_data = \
            _read_data_from(conf['in']['unlabeled'])['review'] \
            if conf['word2vec']['retrain'] else None
        return Word2VecVectorizer(
            conf['word2vec']['strategy'],
            conf['word2vec']['model'],
            train_data=train_data,
            model_args=args,
            averager_args=conf[conf['word2vec']['strategy']])
    else:
        raise Exception("Unknown vectorizer type.")


# NOTE: @Andre, this is the place to add other classifiers.
def _make_classifier(conf):
    _fn = {
        'random-forest': RandomForestClassifier,
    }
    return _fn[conf['classifier']['type']](**conf['classifier']['args'])


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    if conf['run']['type'] == 'optimization':
        train_data = _read_data_from(conf['in']['labeled'])
        ids = np.array(train_data['id'], dtype=np.unicode_)
        reviews = clean_up_reviews(train_data['review'],
                                   conf['run']['remove_stopwords'],
                                   not conf['run']['cache_clean'])
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
                                   not conf['run']['cache_clean'])
        sentiments = np.array(train_data['sentiment'], dtype=np.bool_)
        test_reviews = clean_up_reviews(test_data['review'],
                                   conf['run']['remove_stopwords'],
                                   not conf['run']['cache_clean'])

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
    main()
