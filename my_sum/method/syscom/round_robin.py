# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, unicode_literals, print_function

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

class RoundRobinSummarizer(object):

    def __init__(self, language="english"):
        self.language = language.lower()

        if self.language.startswith("en"):
            self._get_sentence_length = get_en_sentence_length
        if self.language.startswith("ch"):
            self._get_sentence_length = get_cn_sentence_length

    def __call__(self, documentSet, words_limit, summarizer1, summarizer2, summarizer3):
        return self.round_robin(documentSet, words_limit, summarizer1, summarizer2, summarizer3)

    def is_summary(self, sentence, summary_word_count, words_limit):
        if self._get_sentence_length(sentence) <= 5:
            return False
        if self._get_sentence_length(sentence) + summary_word_count <= words_limit:
            return True

    def round_robin(self, document, words_limit, summarizer1, summarizer2, summarizer3):
        summary_1 = summarizer1(document, words_limit+50, summary_order="rating")
        summary_2 = summarizer2(document, words_limit+50, summary_order="rating")
        summary_3 = summarizer3(document, words_limit+50, summary_order="rating")
        length = max(len(summary_1), len(summary_2), len(summary_3))
        summary = []
        summary_word_count = 0
        sentences = document.sentences

        for i in xrange(length):
            if summary_word_count >= words_limit:
                break
            if i <= len(summary_1) - 1 and self.is_summary(summary_1[i], summary_word_count, words_limit) \
               and summary_1[i] not in summary:
                summary.append(summary_1[i])
                summary_word_count += self._get_sentence_length(summary_1[i])
            if i <= len(summary_2) - 1 and self.is_summary(summary_2[i], summary_word_count, words_limit) \
               and summary_2[i] not in summary:
                summary.append(summary_2[i])
                summary_word_count += self._get_sentence_length(summary_2[i])
            if i <= len(summary_3) - 1 and self.is_summary(summary_3[i], summary_word_count, words_limit) \
               and summary_3[i] not in summary:
                summary.append(summary_3[i])
                summary_word_count += self._get_sentence_length(summary_3[i])

        summary = sorted(summary, key=lambda x: sentences.index(x))
        return summary
