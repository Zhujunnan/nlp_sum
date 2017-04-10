# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import math
import numpy
from numpy.linalg import norm

from nltk.stem import SnowballStemmer
from ._summarizer import AbstractSummarizer
from nlp_sum.my_sum.similarity.cosine_sim import compute_tf, compute_idf


class TextRankSummarizer(AbstractSummarizer):

    _stop_words = frozenset()
    epsilon=0.01

    def __init__(self, language="english", stemmer_or_not=False):
        if language.startswith("en") and stemmer_or_not:
            super(TextRankSummarizer, self).__init__(
                language,
                SnowballStemmer("english").stem
            )
        else:
            super(TextRankSummarizer, self).__init__(language)

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, document_set, words_limit, method="default"):
        similarity_matrix = self._create_matrix(document_set)
        scores = self.power_method(similarity_matrix, self.epsilon)
        ratings = dict(zip(document_set.sentences, scores))

        if method.lower() == "mmr":
            dictionary = self._create_dictionary(document_set)
            if not dictionary:
                return ()
            tfidf_matrix = self._create_tfidf_matrix(document_set, dictionary)
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
        Creates matrix of shape |unique words|×|sentences| where cells
        contains number of occurences of words (rows) in senteces (cols).
        """
        sentence_number_every_doc = [len(doc.sentences) for doc in document_set.documents]
        words_in_every_doc = [self._normalize_words(doc.words)
                              for doc in document_set.documents]
        tf_value_every_doc = compute_tf(words_in_every_doc)
        idf_value = compute_idf(words_in_every_doc)

        sentences = document_set.sentences
        words_count = len(dictionary)
        sentences_count = len(sentences)

        # create matrix |unique_words|x|sentences| filled with zeroes
        matrix = numpy.zeros((words_count, sentences_count))
        sent_idx_start = 0
        for doc_idx, sent_num in enumerate(sentence_number_every_doc):
            for idx in xrange(sent_idx_start, sent_num + sent_idx_start):
                for word in self._normalize_words(sentences[idx].words):
                    if word in dictionary:
                        row = dictionary[word]
                        matrix[row, idx] = tf_value_every_doc[doc_idx][word] * idf_value[word]
            sent_idx_start += sent_num

        return matrix

    def _get_similarity(self, word_list1, word_list2):
        words = list(set(word_list1 + word_list2))
        vector1 = [float(word_list1.count(word)) for word in words]
        vector2 = [float(word_list2.count(word)) for word in words]

        vector_cos = [vector1[idx] * vector2[idx] for idx in xrange(len(vector1))]
        cooccur_num = sum([1 for num in vector_cos if num > 0])

        if abs(cooccur_num) <= 1e-7:
            return 0

        denominator = math.log(len(word_list1) * len(word_list2))

        if denominator <= 1e-7:
            return 0

        return cooccur_num / denominator

    def _create_matrix(self, document_set):
        sentences = document_set.sentences
        sentences_count = len(sentences)
        sim_matrix = numpy.zeros((sentences_count, sentences_count))

        for i in xrange(sentences_count):
            for j in xrange(i):
                word_list1 = self._normalize_words(sentences[i].words)
                word_list2 = self._normalize_words(sentences[j].words)
                sim_matrix[i, j] = self._get_similarity(word_list1, word_list2)

        sim_matrix = sim_matrix + sim_matrix.T

        #normalize the matrix by row
        for i in xrange(sentences_count):
            degree = sim_matrix[i, :].sum()
            for j in xrange(sentences_count):
                if degree < 1e-7:
                    continue
                sim_matrix[i, j] = sim_matrix[i, j] / degree

        return sim_matrix

    def power_method(self, matrix, epsilon, para_damp=0.15):
        sentences_count = len(matrix)
        damp_matrix = numpy.ones((sentences_count, sentences_count)) * (1 / sentences_count)
        matrix = para_damp * damp_matrix + (1 - para_damp) * matrix
        transposed_matrix = matrix.T
        residual = float('inf')
        p_vector = numpy.array([1.0 / sentences_count] * sentences_count)

        while residual > epsilon:
            next_p_vector = numpy.dot(transposed_matrix, p_vector)
            residual = norm(numpy.subtract(next_p_vector, p_vector))
            p_vector = next_p_vector

        return p_vector
