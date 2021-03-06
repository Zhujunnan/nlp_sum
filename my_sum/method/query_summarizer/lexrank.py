# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import math
from collections import Counter
from collections import namedtuple
from operator import attrgetter

from ._summarizer import AbstractSummarizer

import numpy
import copy
from numpy.linalg import norm
from nltk.stem import SnowballStemmer
from nlp_sum.my_sum.similarity.cosine_sim import compute_tf, compute_idf
from nlp_sum.test.utils_for_test import build_document_from_string


SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating"))


class LexRank_querySummarizer(AbstractSummarizer):
    """
    LexRank: Graph-based Centrality as Salience in Text Summarization
    Source: http://tangra.si.umich.edu/~radev/lexrank/lexrank.pdf
    """
    thresold = 0.1
    epsilon = 0.1
    _stop_words = frozenset()

    def __init__(self, language="english", stemmer_or_not=False):
        if language.startswith("en") and stemmer_or_not:
            super(LexRank_querySummarizer, self).__init__(
                language,
                SnowballStemmer("english").stem
            )
        else:
            super(LexRank_querySummarizer, self).__init__(language)

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, document_set, query, words_limit, method="mmr", summary_order="origin"):
        self.summary_order = summary_order
        document_query = build_document_from_string(query, self.language)
        sentence_count = len(document_set.sentences)
        query_sent_count = len(document_query.sentences)

        dictionary = self._create_dictionary(document_set, document_query)
        # empty document
        if not dictionary:
            return ()

        tfidf_matrix = self._create_tfidf_matrix(document_set, document_query, dictionary)
        sim_matrix, initial_scores = self._create_sim_matrix(document_set, tfidf_matrix, sentence_count, query_sent_count, self.thresold)
        scores = self.power_method(sim_matrix, initial_scores, self.epsilon)
        ratings = dict(zip(document_set.sentences, scores))
        tfidf_matrix = tfidf_matrix[:, :sentence_count]
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

    def _create_dictionary(self, document_set, document_query):
        """
        Creates mapping key = word, value = row index
        """
        words = self._normalize_words(document_set.words)
        words_query = self._normalize_words(document_query.words)
        words = words + words_query
        unique_words = frozenset(words)
        return dict((word, idx) for idx, word in enumerate(unique_words))

    def _create_tfidf_matrix(self, document_set, document_query, dictionary):
        """
        summarization should treat a sentence as a doc
        Creates matrix of shape |unique words|×|sentences| where cells
        contains number of occurences of words (rows) in senteces (cols).
        """
        sentences_count = len(document_set.sentences) + len(document_query.sentences)
        words_in_every_sent = [self._normalize_words(sent.words)
                               for sent in document_set.sentences]
        words_in_every_sent_query = [self._normalize_words(sent.words)
                                     for sent in document_query.sentences]
        words_in_every_sent = words_in_every_sent + words_in_every_sent_query
        tf_value_every_sent = compute_tf(words_in_every_sent)
        idf_value = compute_idf(words_in_every_sent)

        words_count = len(dictionary)
        # print(document_query.words)
        # create matrix |unique_words|x|sentences| filled with zeroes
        matrix = numpy.zeros((words_count, sentences_count))
        for idx, sentence in enumerate(document_set.sentences + document_query.sentences):
            for word in self._normalize_words(sentence.words):
                if word in dictionary:
                    row = dictionary[word]
                    matrix[row, idx] = tf_value_every_sent[idx][word] * idf_value[word]
        return matrix

    def _create_sim_matrix(self, document_set, tfidf_matrix, sentence_count, query_sent_count, thresold):
        """
        Creates matrix of shape |sentences|×|sentences|.
        """
        # create matrix |sentences|×|sentences| filled with zeroes
        # transpose the matrix so that a row represents a sentence
        similarity_matrix = numpy.zeros((sentence_count, sentence_count))
        degrees = numpy.zeros((sentence_count, ))

        if tfidf_matrix.shape[1] == sentence_count + query_sent_count:
            tfidf_matrix = tfidf_matrix.T

        for sent_num1 in xrange(sentence_count):
            for sent_num2 in xrange(sent_num1):
                similarity_matrix[sent_num1, sent_num2] = self._cosSim(tfidf_matrix[sent_num1, :],
                                                                       tfidf_matrix[sent_num2, :])

        similarity_matrix = similarity_matrix + similarity_matrix.T + numpy.eye(sentence_count)
        initial_scores = numpy.zeros(sentence_count)

        for row in xrange(sentence_count):
            for col in xrange(sentence_count):
                if similarity_matrix[row, col] > thresold:
                    similarity_matrix[row, col] = 1.0
                    degrees[row] += 1
                else:
                    similarity_matrix[row, col] = 0
            for row_query in xrange(sentence_count, sentence_count + query_sent_count):
                initial_scores[row] += self._cosSim(tfidf_matrix[row, :], tfidf_matrix[row_query, :])

        # print(initial_scores)
        query_weight = initial_scores.sum()
        if query_weight != 0:
            initial_scores = initial_scores/query_weight
        else:
            initial_scores = numpy.ones(sentence_count)/sentence_count

        for row in xrange(sentence_count):
            for col in xrange(sentence_count):
                if degrees[row] ==0:
                    degrees[row] = 1
                # now the matrix is not symmetric
                similarity_matrix[row, col] = similarity_matrix[row, col] / degrees[row]


        return similarity_matrix, initial_scores

    @staticmethod
    def power_method(matrix, initial_scores, epsilon, para_d=0.15):
        # :para_d: damping factor, typically chosen in [0.1, 0.2]
        sentences_count = len(matrix)
        damp_matrix = numpy.ones((sentences_count, sentences_count)) * (1 / sentences_count)
        matrix = para_d * damp_matrix + (1 - para_d) * matrix
        transposed_matrix = matrix.T
        residual = float('inf')
        p_vector = initial_scores

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
