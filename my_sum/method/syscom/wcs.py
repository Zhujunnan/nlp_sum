# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, unicode_literals, print_function

import numpy as np
from numpy.linalg import norm

from os.path import abspath, dirname
from nlp_sum.my_sum.parse.plaintext import PlaintextParser
from nlp_sum.my_sum.method.extract_summarizer.submodular import SubmodularSummarizer
from nlp_sum.my_sum.method.extract_summarizer.lexrank import LexRankSummarizer
from nlp_sum.my_sum.method.extract_summarizer.lsa import LsaSummarizer

from nlp_sum.my_sum.similarity.cosine_sim import compute_tf, compute_idf
from nlp_sum.my_sum.similarity.cosine_sim import cosine_similarity
from nlp_sum.my_sum.utils import to_unicode

from nlp_sum.test.utils_for_test import get_cn_sentence_length, get_en_sentence_length

# summarizer_en1 = SubmodularSummarizer("english")
# summarizer_en2 = LexRankSummarizer("english")
# summarizer_en3 = LsaSummarizer("english")

class WcsSummarizer(object):
    LIMIT = float('inf')

    def __init__(self, language="english"):
        """
        weighted consensus multi-document summarization
        """
        self.language = language.lower()

        if self.language.startswith("en"):
            self._get_sentence_length = get_en_sentence_length
        if self.language.startswith("ch"):
            self._get_sentence_length = get_cn_sentence_length

    def __call__(self, documentSet, words_limit, summarizer1, summarizer2, summarizer3):
        return self.weighted_rank(documentSet, words_limit, summarizer1, summarizer2, summarizer3)

    @staticmethod
    def theta(v, z):
        """
        efficient projections onto the l1-ball for learning in high dimensions
        """
        v = v.tolist()[0]
        v = sorted(v, reverse=True)
        length = len(v)

        n = 0
        for i in xrange(length - 1, -1, -1):
            all_sum = sum([v[j] for j in xrange(0, i+1)])
            if v[i] - (all_sum - z)/(i + 1) > 0:
                n = i
                break
        all_sum = sum([v[k] for k in xrange(n+1)])
        theta = (all_sum - z)/(n + 1)
        return theta

    @staticmethod
    def wcs(rank_list, LAMBDA=0.999):
        number = len(rank_list)
        w = [1/number for i in xrange(number)]
        w = np.mat(w)
        r = np.mat(rank_list)

        while True:
            w_old = w
            r_star = w * r
            delta = r_star - r
            d = np.mat([norm(i) * norm(i) for i in delta])
            new_d = d * (LAMBDA - 1) / (2*LAMBDA)
            theta_new = WcsSummarizer.theta(new_d, 1)
            w = new_d - theta_new
            for i in xrange(number):
                w[0, i] = max(0, w[0,i])

            if norm(w - w_old) < 0.1:
                return (w * r).tolist()[0]

    def weighted_rank(self, document, words_limit, summarizer1, summarizer2, summarizer3):
        summary_1 = summarizer1(document, words_limit+50, summary_order="rating")
        summary_2 = summarizer2(document, words_limit+50, summary_order="rating")
        summary_3 = summarizer3(document, words_limit+50, summary_order="rating")
        sent_count = len(document.sentences)
        sentences = document.sentences

        rank = {}
        rank1 = [sent_count for i in sentences]
        rank2 = [sent_count for i in sentences]
        rank3 = [sent_count for i in sentences]
        for idx, sentence in enumerate (sentences):
            if sentence in summary_1:
                rank1[idx] = summary_1.index(sentence) + 1
            if sentence in summary_2:
                rank2[idx] = summary_2.index(sentence) + 1
            if sentence in summary_3:
                rank3[idx] = summary_3.index(sentence) + 1
        rank_list = [rank1, rank2, rank3]
        rank_list = WcsSummarizer.wcs(rank_list)

        for idx, score in enumerate(rank_list):
            rank[idx] = rank_list[idx]
        ratings = sorted(rank.iteritems(), key=lambda x: x[1])
        summary = []
        summary_word_count = 0

        for item in ratings:
            sentence = sentences[item[0]]
            sentence_length = self._get_sentence_length(sentence)
            if summary_word_count >= words_limit:
                break
            if sentence_length <= 5:
                continue
            if sentence_length + summary_word_count <= words_limit:
                summary.append((item[0], sentence))
                summary_word_count += sentence_length

        summary = sorted(summary, key=lambda x: x[0])
        summary_idx, summary_sent = zip(*summary)
        return summary_sent
