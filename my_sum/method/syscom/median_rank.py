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

class MedianRankSummarizer(object):
    LIMIT = float('inf')

    def __init__(self, language="english"):
        self.language = language.lower()

        if self.language.startswith("en"):
            self._get_sentence_length = get_en_sentence_length
        if self.language.startswith("ch"):
            self._get_sentence_length = get_cn_sentence_length

    def __call__(self, documentSet, words_limit, summarizer1, summarizer2, summarizer3):
        return self.median_rank(documentSet, words_limit, summarizer1, summarizer2, summarizer3)

    def median_rank(self, document, words_limit, summarizer1, summarizer2, summarizer3):
        summary_1 = summarizer1(document, self.LIMIT, summary_order="rating")
        summary_2 = summarizer2(document, self.LIMIT, summary_order="rating")
        summary_3 = summarizer3(document, self.LIMIT, summary_order="rating")
        # summary_text1 = ' '.join(sentence._texts for sentence in summary_1)
        # summary_text2 = ' '.join(sentence._texts for sentence in summary_2)
        # summary_text3 = ' '.join(sentence._texts for sentence in summary_3)

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

        for idx, rating in enumerate(zip(rank1, rank2, rank3)):
            rank[idx] = sorted(rating)[1]
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
