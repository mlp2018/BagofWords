#!/usr/bin/env python

import re
import nltk
import warnings

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

# Turn off warnings about Beautiful Soup (Johanna has checked all of them manually)
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

class KaggleWord2VecUtility(object):
    """KaggleWord2VecUtility is a utility class for processing raw HTML text into segments for further learning"""

    @staticmethod
    def review_to_wordlist(review : str, remove_stopwords : bool = False):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
        # 1. Remove HTML
        review_text = BeautifulSoup(review, 'html.parser').get_text()
        #
        # 2. Remove URLs/links by means of a regular expression.
        review_text = re.sub("(www.|http[s]?:\/)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", review_text)
        #
        # 3. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 4. Convert words to lower case and split them
        words = review_text.lower().split()
        #
        # 5. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #
        # 6. Return a list of words
        return(words)

    # Define a function to split a review into parsed sentences
    @staticmethod
    def review_to_sentences(review : bytes, tokenizer,
        remove_stopwords : bool = False ):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #
        # 1. Use the NLTK tokenizer to split the paragraph into sentences
        raw_sentences = tokenizer.tokenize(review.strip())
        #
        # 2. Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append(KaggleWord2VecUtility.review_to_wordlist(raw_sentence, \
                  remove_stopwords))
        #
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences
