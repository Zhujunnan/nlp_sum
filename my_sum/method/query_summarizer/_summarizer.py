# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from collections import namedtuple
from operator import attrgetter
import math

import re
import numpy
from numpy.linalg import norm

from nlp_sum.my_sum.utils import to_unicode, cached_property


def null_stemmer(string):
    "Converts given string to unicode with lower letters."
    return to_unicode(string).lower()

SentenceInfo = namedtuple("SentenceInfo", ("sentence", "order", "rating"))

def get_cn_sentence_length(sentence):
    """
    get the actual length of chinese sentence
    :para : Sentence()
    """
    # the length of ', NBA' should be two
    # the length of ',NBA' will be one
    # the same behavior as microsoft word
    chinese_word_pattern = re.compile(u"[\u4e00-\u9fa5。；，：“”（）、？《》]+",
                                      re.UNICODE)
    english_or_number_pattern = re.compile(u"[^\u4e00-\u9fa5\s。；，：“”（）、？《》]+",
                                           re.UNICODE)
    chinese_word_list = re.findall(chinese_word_pattern, sentence._texts)
    english_or_number_list = re.findall(english_or_number_pattern, sentence._texts)
    chinese_len = len(''.join(chinese_word_list))
    english_or_number_len = len(english_or_number_list)
    # 1 represents the '。'
    return chinese_len + english_or_number_len + 1

def get_en_sentence_length(sentence):
    words_list = sentence._texts.split()
    return len(words_list)

class AbstractSummarizer(object):
    """
    summary_order:
    origin: sentence appear as same as the order in the original doc
    rating: sentence appear in the order of the rating(descending order)
    """
    _summary_order = "origin"

    def __init__(self, language="english", stemmer=null_stemmer):
        if not callable(stemmer):
            raise ValueError("Stemmer should be a callable method")

        self._stemmer = stemmer
        self.language = language.lower()

        if self.language.startswith("en"):
            self._get_sentence_length = get_en_sentence_length
        if self.language.startswith("ch"):
            # calculate the length of the chinese sentence
            # 1 represents the`。`
            self._get_sentence_length = get_cn_sentence_length

    def __call__(self, input_document, words_count):
        raise NotImplementedError("The method should be overridden in subclass")

    @property
    def summary_order(self):
        return self._summary_order

    @summary_order.setter
    def summary_order(self, sum_order):
        self._summary_order = sum_order

    def normalize_word(self, word):
        return self._stemmer(to_unicode(word).lower())

    def _cosSim(self, vector1, vector2):
        # both vector are row vectors
        vector1, vector2 = numpy.mat(vector1), numpy.mat(vector2)
        numerator = float(vector1 * vector2.T)
        denominator = norm(vector1) * norm(vector2)
        if denominator > 0:
            return numerator / denominator
        else:
            return 0.0

    def _get_best_sentences(self, sentences, word_limit, ratings, *args, **kwargs):
        rate = ratings
        if isinstance(ratings, dict):
            assert not args and not kwargs
            rate = lambda sent : ratings[sent]

        # infos is a generator
        infos = (
            SentenceInfo(sentence, order, rate(sentence, *args, **kwargs))
            for order, sentence in enumerate(sentences)
        )

        # sort sentences by rating in descending order
        infos = sorted(infos, key=attrgetter("rating"), reverse=True)
        summary = []
        summary_word_count = 0

        for info in infos:
            sentence_length = self._get_sentence_length(info.sentence)
            if summary_word_count >= word_limit:
                break
            if sentence_length <= 8:
                continue
            # alow the length of summary exceed little than limit
            if sentence_length + summary_word_count <= word_limit:
                # summary.append(info.sentence)
                summary.append(info)
                summary_word_count += sentence_length

        if self.summary_order == "origin":
            summary = sorted(summary, key=attrgetter("order"))

        return tuple(info.sentence for info in summary)

    def _get_best_sentences_by_MMR(self, sentences, words_limit,
                                   matrix, ratings, para=0.7, beta=0.1):
        """
        :para matrix: tfidf matrix in numpy format
        :para ratings: sentence score :type dict or generator
        """
        rate = ratings
        if isinstance(ratings, dict):
            rate = lambda sent: ratings[sent]

        infos = [SentenceInfo(sent, order, rate(sent)) for order, sent in enumerate(sentences)]
        # infos = sorted(infos, key=attrgetter("rating"), reverse=True)
        sent_chosen = [False for i in sentences]
        summary_word_count = 0
        summary = []
        # transpose the matrix so that a row represent a sentence
        sent_matrix = matrix.T

        # first chose the best sentence to add to summary
        first_sent_idx = -1
        first_max_score = float('-inf')
        for info in infos:
            sent_len = self._get_sentence_length(info.sentence)
            if sent_len < words_limit and sent_len > 5:
                sent_score_punish_len = info.rating / math.pow(sent_len, beta)
                if sent_score_punish_len > first_max_score:
                    first_max_score = sent_score_punish_len
                    first_sent_idx = info.order
        if first_sent_idx < 0:
            return []
        summary.append(infos[first_sent_idx])
        sent_chosen[first_sent_idx] = True
        summary_word_count += self._get_sentence_length(infos[first_sent_idx].sentence)

        while summary_word_count < words_limit:
            max_score = float('-inf')
            pick_sent_idx = -1
            for info in infos:
                if sent_chosen[info.order]:
                    continue
                tmp_score = info.rating
                sent_length = self._get_sentence_length(info.sentence)
                for sent in summary:
                    info_vector = sent_matrix[info.order, :]
                    sent_vector = sent_matrix[sent.order, :]
                    similarity = self._cosSim(info_vector, sent_vector)

                    score_new = info.rating - similarity * sent.rating * para
                    if score_new < tmp_score:
                        tmp_score = score_new

                    score_punish_len = tmp_score / math.pow(sent_length, beta)
                    if score_punish_len > max_score and not sent_chosen[info.order] and \
                       sent_length + summary_word_count < words_limit and sent_length > 5 :
                        max_score = score_punish_len
                        pick_sent_idx = info.order

            if pick_sent_idx == -1:
                break
            summary_word_count += self._get_sentence_length(infos[pick_sent_idx].sentence)
            if summary_word_count >= words_limit:
                break
            sent_chosen[pick_sent_idx] = True
            summary.append(infos[pick_sent_idx])
        if self.summary_order == "origin":
            summary = sorted(summary, key=attrgetter("order"))
        return tuple(info.sentence for info in summary)
