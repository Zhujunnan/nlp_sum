# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import numpy
import math
from numpy import *
from time import time
from collections import namedtuple
from numpy.linalg import norm
from nltk.stem import SnowballStemmer

from ._summarizer import AbstractSummarizer
from nlp_sum.my_sum.similarity.cosine_sim import compute_tf, compute_idf
from nlp_sum.my_sum.similarity.cosine_sim import cosine_similarity

SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating"))


class NmfSummarizer(AbstractSummarizer):
    _stop_words = frozenset()

    def __init__(self, language="english", stemmer_or_not=False):
        if language.startswith("en") and stemmer_or_not:
            super(NmfSummarizer, self).__init__(
                language,
                SnowballStemmer("english").stem
            )
        else:
            super(NmfSummarizer, self).__init__(language)

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, documentSet, words_limit, method="mmr", metric="tf", r=None):
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
        ranks = iter(self._compute_ranks(matrix, r))

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
                "NMF algorithm may not work properly."
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
                      + "NMF algorithm may not work properly."
            warn(message.format(words_count, sentences_count))
            # create matrix |unique_words|x|sentences| filled with zeroes
        matrix = numpy.zeros((words_count, sentences_count))
        for idx, sentence in enumerate(document_set.sentences):
            for word in self._normalize_words(sentence.words):
                if word in dictionary:
                    row = dictionary[word]
                    matrix[row, idx] = tf_value_every_sent[idx][word] * idf_value[word]
        return matrix

    def _compute_ranks(self, tfidf_matrix, r_number=None):
        """
        tfidf_matrix: m * n matrix =====> (m * r) * (r * n)
        """
        sent_count = tfidf_matrix.shape[1]
        matrix_rank = len(tfidf_matrix)
        if not r_number:
            r_number = math.ceil(matrix_rank / 2)
            r_number = int(r_number)
        matrix_order = r_number

        origin_w = numpy.random.randint(0, 9, size=(matrix_rank, matrix_order))
        origin_h = numpy.random.randint(0, 9, size=(matrix_order, sent_count))
        v = numpy.dot(origin_w, origin_h)

        w = numpy.random.randint(0, 9, size=(matrix_rank, matrix_order))
        h = numpy.random.randint(0, 9, size=(matrix_order, sent_count))

        (output_W, output_H) = self.nmf(v, w, h, 0.001, 50, 100)

        ranks = []
        h_sum = output_H.sum()
        h_row_sum = [row.sum() for row in output_H]
        for i in xrange(sent_count):
            rank = sum(output_H[j, i] * h_row_sum[j] for j in xrange(r_number))
            ranks.append(rank/h_sum)
        return ranks

    @staticmethod
    def nmf(V, Winit, Hinit, tol, timelimit, maxiter):
        """
            (W,H) = nmf(V, Winit, Hinit, tol, timelimit, maxiter)
            W,H: output solution
            Winit,Hinit: initial solution
            tol: tolerance for a relative stopping condition
            timelimit, maxiter: limit of time and iterations
        """

        W = Winit; H = Hinit; initt = time();

        gradW = dot(W, dot(H, H.T)) - dot(V, H.T)
        gradH = dot(dot(W.T, W), H) - dot(W.T, V)

        initgrad = norm(r_[gradW, gradH.T])
        tolW = max(0.001, tol) * initgrad
        tolH = tolW

        for iter in xrange(1,maxiter):
            # stopping condition
            projnorm = norm(r_[gradW[logical_or(gradW<0, W>0)],
                               gradH[logical_or(gradH<0, H>0)]])
            if projnorm < tol*initgrad or time() - initt > timelimit: break

            (W, gradW, iterW) = NmfSummarizer.nlssubprob(V.T, H.T, W.T, tolW, 1000)
            W, gradW = W.T, gradW.T

            if iterW==1: tolW = 0.1 * tolW

            (H,gradH,iterH) = NmfSummarizer.nlssubprob(V, W, H, tolH, 1000)
            if iterH==1: tolH = 0.1 * tolH

        # print '\nIter = %d Final proj-grad norm %f' % (iter, projnorm)
        return (W, H)

    @staticmethod
    def nlssubprob(V, W, Hinit, tol, maxiter):
        """
            H, grad: output solution and gradient
            iter: #iterations used
            V, W: constant matrices
            Hinit: initial solution
            tol: stopping tolerance
            maxiter: limit of iterations
        """

        H = Hinit
        WtV = dot(W.T, V)
        WtW = dot(W.T, W)

        alpha = 1; beta = 0.1;
        for iter in xrange(1, maxiter):
            grad = dot(WtW, H) - WtV
            projgrad = norm(grad[logical_or(grad < 0, H >0)])
            if projgrad < tol: break

            # search step size
            for inner_iter in xrange(1,20):
                Hn = H - alpha*grad
                Hn = where(Hn > 0, Hn, 0)
                d = Hn-H
                gradd = sum(grad * d)
                dQd = sum(dot(WtW,d) * d)
                suff_decr = 0.99*gradd + 0.5*dQd < 0;
                if inner_iter == 1:
                    decr_alpha = not suff_decr; Hp = H;
                if decr_alpha:
                    if suff_decr:
                        H = Hn; break;
                    else:
                        alpha = alpha * beta;
                else:
                    if not suff_decr or (Hp == Hn).all():
                        H = Hp; break;
                    else:
                        alpha = alpha/beta; Hp = Hn;

            # if iter == maxiter:
            #     print 'Max iter in nlssubprob'
        return (H, grad, iter)
