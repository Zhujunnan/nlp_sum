# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import random

from ._summarizer import AbstractSummarizer


class RandomSummarizer(AbstractSummarizer):
    """pick sentences randomly"""

    def __init__(self, language="english", stemmer_or_not=False):
        super(RandomSummarizer, self).__init__(language)

    def __call__(self, document_set, words_limit, summary_order="origin"):
        self.summary_order = summary_order
        sentences = document_set.sentences
        ratings = self._get_random_ratings(sentences)
        return self._get_best_sentences(sentences, words_limit, ratings)

    def _get_random_ratings(self, sentences):
        ratings = range(len(sentences))
        random.shuffle(ratings)

        return dict((sent, rating) for sent, rating in zip(sentences, ratings))
