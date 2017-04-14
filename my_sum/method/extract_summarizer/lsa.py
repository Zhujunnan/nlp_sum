# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import math
from warnings import warn

import numpy
from collections import namedtuple
from operator import attrgetter
from numpy.linalg import svd, norm
from nltk.stem import SnowballStemmer

from ._summarizer import AbstractSummarizer
from nlp_sum.my_sum.similarity.cosine_sim import compute_tf, compute_idf
from nlp_sum.my_sum.similarity.cosine_sim import cosine_similarity


SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating"))


class LsaSummarizer(AbstractSummarizer):
    MIN_DIMENSIONS = 3
    REDUCTION_RATIO = 1/1
    _stop_words = frozenset()

    def __init__(self, language="english", stemmer_or_not=False):
        if language.startswith("en") and stemmer_or_not:
            super(LsaSummarizer, self).__init__(
                language,
                SnowballStemmer("english").stem
            )
        else:
            super(LsaSummarizer, self).__init__(language)

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    # def __call__(self, documentSet, words_limit, method="default"):
    #     dictionary = self._create_dictionary(documentSet)
    #     # empty document
    #     if not dictionary:
    #         return ()

    #     matrix = self._create_tfidf_matrix(documentSet, dictionary)
    #     u, sigma, v = svd(matrix, full_matrices=False)
    #     ranks = iter(self._compute_ranks(sigma, v))
    #     if method.lower() == "default":
    #         return self._get_best_sentences(documentSet.sentences, words_limit,
    #                                         lambda sent: next(ranks))
    #     if method.lower() == "mmr":
    #         return self._get_best_sentences_by_MMR(documentSet.sentences, words_limit,
    #                                                matrix, lambda sent: next(ranks))

    def __call__(self, documentSet, words_limit, method="mmr", metric="tf"):
        dictionary = self._create_dictionary(documentSet)
        # empty document
        if not dictionary:
            return ()
        if metric.lower() == "tf":
            matrix = self._create_matrix(documentSet, dictionary)
            matrix = self._compute_term_frequency(matrix)
        elif metric.lower() == "tfidf":
            matrix = self._create_tfidf_matrix(documentSet, dictionary)
        else:
            raise ValueError("Don't support your metric now.")
        u, sigma, v = svd(matrix, full_matrices=False)
        ranks = iter(self._compute_ranks(sigma, v))

        if method.lower() == "default":
            return self._get_best_sentences(documentSet.sentences, words_limit,
                                            lambda sent: next(ranks))
        if method.lower() == "mmr":
            return self._get_best_sentences_by_MMR(documentSet.sentences, words_limit,
                                                   matrix, lambda sent: next(ranks))

    def _filter_out_stop_words(self, words):
        return [word for word in words if word not in self._stop_words]

    def _normalize_words(self, words):
        words = map(self.normalize_word, words)
        return self._filter_out_stop_words(words)

    def _get_content_words_in_sentence(self, sentence):
        normalized_content_words = self._normalize_words(sentence.words)
        return normalized_content_words

    def _get_all_content_words_in_doc(self, sentences):
        all_words = [word for sent in sentences for word in sent.words]
        normalized_content_words = self._normalize_words(all_words)
        return normalized_content_words

    def _create_dictionary(self, document):
        """
        Creates mapping key = word, value = row index
        """
        words = self._normalize_words(document.words)
        unique_words = frozenset(words)
        return dict((word, idx) for idx, word in enumerate(unique_words))

    def _create_matrix(self, document, dictionary):
        """
        Creates matrix of shape |unique words|×|sentences| where cells
        contains number of occurences of words (rows) in senteces (cols).
        """
        sentences = document.sentences

        words_count = len(dictionary)
        sentences_count = len(sentences)
        if words_count < sentences_count:
            message = (
                "Number of words (%d) is lower than number of sentences (%d). "
                "LSA algorithm may not work properly."
            )
            warn(message % (words_count, sentences_count))

        # create matrix |unique words|×|sentences| filled with zeroes
        matrix = numpy.zeros((words_count, sentences_count))
        for col, sentence in enumerate(sentences):
            for word in self._normalize_words(sentence.words):
                # only valid words is counted (not stop-words, ...)
                if word in dictionary:
                    row = dictionary[word]
                    matrix[row, col] += 1

        return matrix

    def _compute_term_frequency(self, matrix, smooth=0.4):
        """
        Computes TF metrics for each sentence (column) in the given matrix.
        You can read more about smoothing parameter at URL below:
        http://nlp.stanford.edu/IR-book/html/htmledition/maximum-tf-normalization-1.html
        """
        assert 0.0 <= smooth < 1.0

        max_word_frequencies = numpy.max(matrix, axis=0)
        rows, cols = matrix.shape
        for row in range(rows):
            for col in range(cols):
                max_word_frequency = max_word_frequencies[col]
                if max_word_frequency != 0:
                    frequency = matrix[row, col]/max_word_frequency
                    matrix[row, col] = smooth + (1.0 - smooth)*frequency

        return matrix

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
        if words_count < sentences_count:
            message = "Number of words {0} is smaller than number of sentences {1}." \
                      + "LSA algorithm may not work properly."
            warn(message.format(words_count, sentences_count))
        # create matrix |unique_words|x|sentences| filled with zeroes
        matrix = numpy.zeros((words_count, sentences_count))
        for idx, sentence in enumerate(document_set.sentences):
            for word in self._normalize_words(sentence.words):
                if word in dictionary:
                    row = dictionary[word]
                    matrix[row, idx] = tf_value_every_sent[idx][word] * idf_value[word]
        return matrix

    def _compute_ranks(self, sigma, v_matrix):
        assert len(sigma) == v_matrix.shape[0], "Matrices must be multiplicable"

        dimensions = max(LsaSummarizer.MIN_DIMENSIONS,
                         int(len(sigma)*LsaSummarizer.REDUCTION_RATIO))
        sigma_square = tuple(s**2 if idx < dimensions else 0.0
                             for idx, s in enumerate(sigma))

        ranks = []
        # iterate over columns of matrix (rows of transposed matrix)
        for column_vector in v_matrix.T:
            rank = sum(s*v**2 for s, v in zip(sigma_square, column_vector))
            ranks.append(math.sqrt(rank))
        return ranks

    # def _create_tfidf_matrix(self, document_set, dictionary):
    #     """
    #     Creates matrix of shape |unique words|×|sentences| where cells
    #     contains number of occurences of words (rows) in senteces (cols).
    #     """
    #     sentence_number_every_doc = [len(doc.sentences) for doc in document_set.documents]
    #     words_in_every_doc = [self._normalize_words(doc.words)
    #                           for doc in document_set.documents]
    #     tf_value_every_doc = compute_tf(words_in_every_doc)
    #     idf_value = compute_idf(words_in_every_doc)

    #     sentences = document_set.sentences
    #     words_count = len(dictionary)
    #     sentences_count = len(sentences)

    #     if words_count < sentences_count:
    #         message = "Number of words {0} is smaller than number of sentences {1}." \
        #                 + "LSA algorithm may not work properly."
    #         warn(message.format(words_count, sentences_count))

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
