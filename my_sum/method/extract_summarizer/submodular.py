# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from ._summarizer import AbstractSummarizer
from nlp_sum.my_sum.similarity.cosine_sim import compute_tf, compute_idf

import math
import numpy
from numpy.linalg import norm
from nltk.stem import SnowballStemmer



class SubmodularSummarizer(AbstractSummarizer):

    _stop_words = frozenset()

    def __init__(self, language="english", stemmer_or_not=False):
        if language.startswith("en") and stemmer_or_not:
            super(SubmodularSummarizer, self).__init__(
                language,
                SnowballStemmer("english").stem
            )
        else:
            super(SubmodularSummarizer, self).__init__(language)

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, document_set, words_limit, Lambda=0.5, beta=0.1):
        assert Lambda < 1, "Lambda must be in interval [0, 1]"
        assert beta < 1, "beta must be in interval [0, 1]"
        return self.greedy(document_set, words_limit, Lambda, beta)

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

    def _create_sim_matrix(self, document_set, tfidf_matrix):
        sentences_count = len(document_set.sentences)
        similarity_matrix = numpy.zeros((sentences_count, sentences_count))

        if tfidf_matrix.shape[1] == sentences_count:
            tfidf_matrix = tfidf_matrix.T

        for sent_num1 in xrange(sentences_count):
            for sent_num2 in xrange(sentences_count):
                similarity_matrix[sent_num1, sent_num2] = self._cosSim(tfidf_matrix[sent_num1, :],
                                                                       tfidf_matrix[sent_num2, :])

        similarity_matrix = similarity_matrix + similarity_matrix.T + numpy.eye(sentences_count)
        return similarity_matrix

    def _row_similarity_sum(self, similarity_matrix):
        sentences_count = len(similarity_matrix)
        # subtract the similarity with sentence self
        return sum(similarity_matrix - numpy.eye(sentences_count))

    def _submodular_L(self, document_set, summary_idx_set, row_similarity_sum,
                      similarity_matrix, sent_idx, alpha):
        # sent_idx represents the sentence which will be added to summary
        # to calculate its score
        score = 0.0
        sentences_count = len(document_set.sentences)

        for sent_idx_doc in xrange(sentences_count):
            # sent_idx will be added to summary
            if sent_idx_doc == sent_idx:
                continue
            sum_similarity = 0
            for sent_idx_summary in summary_idx_set:
                if sent_idx_doc != sent_idx_summary:
                    sum_similarity += similarity_matrix[sent_idx_doc, sent_idx_summary]

                if sent_idx != -1:
                    # add the sentence to summary
                    sum_similarity += similarity_matrix[sent_idx_doc, sent_idx]

            score_final = min(row_similarity_sum[sent_idx_doc] * alpha, sum_similarity)
            score = score + score_final
        return score

    # calculate the redundancy
    def _submodular_R(self, summary_idx_set, similarity_matrix, sent_idx):
        score = 0.0
        for summary_idx in summary_idx_set:
            score = score + similarity_matrix[summary_idx, sent_idx]

        return -score

    def greedy(self, document_set, words_limit, Lambda=0.5, beta=0.1):
        dictionary = self._create_dictionary(document_set)

        if not dictionary:
            return ()

        summary = []
        summary_idx_set = []
        summary_word_count = 0
        sentences = document_set.sentences
        sentences_count = len(sentences)
        alpha = (1 / sentences_count) * 10
        sent_chosen = [False for i in document_set.sentences]

        tfidf_matrix = self._create_tfidf_matrix(document_set, dictionary)
        similarity_matrix = self._create_sim_matrix(document_set, tfidf_matrix)
        row_similarity_sum = self._row_similarity_sum(similarity_matrix)

        while True:
            max_increment_score = float("-inf")
            max_sent_idx = -1
            # score before sentence i being added in to summary
            init_score = self._submodular_L(document_set, summary_idx_set, row_similarity_sum,
                                            similarity_matrix, -1, alpha)

            for sent_idx in xrange(sentences_count):
                sent_len = self._get_sentence_length(sentences[sent_idx])
                if not sent_chosen[sent_idx] and (sent_len + summary_word_count < words_limit) and \
                   sent_len > 5:
                    L_score = self._submodular_L(document_set, summary_idx_set, row_similarity_sum,
                                                 similarity_matrix, sent_idx, alpha)
                    R_score = self._submodular_R(summary_idx_set, similarity_matrix, sent_idx)
                    increment_score = Lambda * L_score + (1 - Lambda) * R_score - Lambda * init_score
                    increment_score = increment_score / math.pow(sent_len, beta)

                    if increment_score > max_increment_score:
                        max_increment_score = increment_score
                        max_sent_idx = sent_idx

            if max_sent_idx == -1:
                break
            summary_word_count += self._get_sentence_length(sentences[max_sent_idx])
            if summary_word_count > words_limit:
                break
            sent_chosen[max_sent_idx] = True
            summary_idx_set.append(max_sent_idx)

        summary_idx_set = sorted(summary_idx_set)
        summary = [sentences[idx] for idx in summary_idx_set]

        return summary
