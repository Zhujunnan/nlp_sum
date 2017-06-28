# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import math
from collections import Counter
from collections import namedtuple
from operator import attrgetter

from ._summarizer import AbstractSummarizer

import numpy
from numpy.linalg import norm
from nltk.stem import SnowballStemmer
from nlp_sum.my_sum.similarity.cosine_sim import compute_tf, compute_idf


SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating"))


class LexRankSummarizer(AbstractSummarizer):
    """
    LexRank: Graph-based Centrality as Salience in Text Summarization
    Source: http://tangra.si.umich.edu/~radev/lexrank/lexrank.pdf
    """
    thresold = 0.1
    epsilon = 0.1
    _stop_words = frozenset()

    def __init__(self, language="english", stemmer_or_not=False):
        if language.startswith("en") and stemmer_or_not:
            super(LexRankSummarizer, self).__init__(
                language,
                SnowballStemmer("english").stem
            )
        else:
            super(LexRankSummarizer, self).__init__(language)

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, document_set, words_limit, method="mmr", summary_order="origin"):
        dictionary = self._create_dictionary(document_set)
        self.summary_order = summary_order
        # empty document
        if not dictionary:
            return ()
        tfidf_matrix = self._create_tfidf_matrix(document_set, dictionary)
        sim_matrix = self._create_sim_matrix(document_set, tfidf_matrix, self.thresold)
        scores = self.power_method(sim_matrix, self.epsilon)
        ratings = dict(zip(document_set.sentences, scores))
        if method.lower() == "mmr":
            return self._get_best_sentences_by_MMR(document_set.sentences, words_limit,
                                                   tfidf_matrix, ratings)
        if method.lower() == "default":
            return self._get_best_sentences(document_set.sentences, words_limit, ratings)

    def _filter_out_stop_words(self, words):
        return [word for word in words if word not in self._stop_words]

    def _normalize_words(self, words):
        words = map(self.normalize_word, words)
        return self._filter_out_stop_words(words)

    def _create_dictionary(self, document_set):
        """
        Creates mapping key = word, value = row index
        """
        words = self._normalize_words(document_set.words)
        unique_words = frozenset(words)
        return dict((word, idx) for idx, word in enumerate(unique_words))

    def _create_tfidf_matrix(self, document_set, dictionary):
        """
        summarization should treat a sentence as a doc
        Creates matrix of shape |unique words|×|sentences| where cells
        contains number of occurences of words (rows) in senteces (cols).
        """
        sentences_count = len(document_set.sentences)
        words_in_every_sent = [self._normalize_words(sent.words)
                               for sent in document_set.sentences]
        tf_value_every_sent = compute_tf(words_in_every_sent)
        idf_value = compute_idf(words_in_every_sent)

        words_count = len(dictionary)
        # create matrix |unique_words|x|sentences| filled with zeroes
        matrix = numpy.zeros((words_count, sentences_count))
        for idx, sentence in enumerate(document_set.sentences):
            for word in self._normalize_words(sentence.words):
                if word in dictionary:
                    row = dictionary[word]
                    matrix[row, idx] = tf_value_every_sent[idx][word] * idf_value[word]
        return matrix

    def _create_sim_matrix(self, document_set, tfidf_matrix, thresold):
        """
        Creates matrix of shape |sentences|×|sentences|.
        """
        # create matrix |sentences|×|sentences| filled with zeroes
        sentences_count = len(document_set.sentences)
        # transpose the matrix so that a row represents a sentence
        similarity_matrix = numpy.zeros((sentences_count, sentences_count))
        degrees = numpy.zeros((sentences_count, ))

        if tfidf_matrix.shape[1] == sentences_count:
            tfidf_matrix = tfidf_matrix.T

        for sent_num1 in xrange(sentences_count):
            for sent_num2 in xrange(sent_num1):
                similarity_matrix[sent_num1, sent_num2] = self._cosSim(tfidf_matrix[sent_num1, :],
                                                                       tfidf_matrix[sent_num2, :])

        similarity_matrix = similarity_matrix + similarity_matrix.T + numpy.eye(sentences_count)
        for row in xrange(sentences_count):
            for col in xrange(sentences_count):
                if similarity_matrix[row, col] > thresold:
                    similarity_matrix[row, col] = 1.0
                    degrees[row] += 1
                else:
                    similarity_matrix[row, col] = 0

        for row in xrange(sentences_count):
            for col in xrange(sentences_count):
                if degrees[row] ==0:
                    degrees[row] = 1
                # now the matrix is not symmetric
                similarity_matrix[row, col] = similarity_matrix[row, col] / degrees[row]

        return similarity_matrix

    @staticmethod
    def power_method(matrix, epsilon, para_d=0.15):
        # :para_d: damping factor, typically chosen in [0.1, 0.2]
        sentences_count = len(matrix)
        damp_matrix = numpy.ones((sentences_count, sentences_count)) * (1 / sentences_count)
        matrix = para_d * damp_matrix + (1 - para_d) * matrix
        transposed_matrix = matrix.T
        residual = float('inf')
        p_vector = numpy.array([1.0 / sentences_count] * sentences_count)

        while residual > epsilon:
            next_p_vector = numpy.dot(transposed_matrix, p_vector)
            residual = norm(numpy.subtract(next_p_vector, p_vector))
            p_vector = next_p_vector

        return p_vector

    # def _create_tfidf_matrix(self, document_set, dictionary):
    #     """
    #     Creates matrix of shape |unique words|×|sentences| where cells
    #     contains number of occurences of words (rows) in senteces (cols).
    #     """
    #     sentence_number_every_doc = [len(doc.sentences) for doc in document_set.documents]
    #     words_in_every_doc = [self._normalize_words(doc.words)
    #                           for doc in document_set.documents]
    #     tf_value_every_doc = compute_tf(words_in_every_doc)


    #     sentences = document_set.sentences
    #     words_count = len(dictionary)
    #     sentences_count = len(sentences)

    #     # create matrix |unique_words|x|sentences| filled with zeroes
    #     matrix = numpy.zeros((words_count, sentences_count))
    #     sent_idx_start = 0
    #     for doc_idx, sent_num in enumerate(sentence_number_every_doc):
    #         for idx in xrange(sent_idx_start, sent_num + sent_idx_start):
    #             for word in self._normalize_words(sentences[idx].words):
    #                 if word in dictionary:
    #                     row = dictionary[word]
    #                     matrix[row, idx] = tf_value_every_doc[doc_idx][word] * idf_value[word]
    #         sent_idx_start += sent_num

    #     return matrix
