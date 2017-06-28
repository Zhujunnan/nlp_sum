# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, unicode_literals

import unittest

from os.path import abspath, dirname
from nlp_sum.my_sum.utils import get_stop_words
from nlp_sum.my_sum.parse.plaintext import PlaintextParser
from nlp_sum.my_sum.method.extract_summarizer.kl import KLSummarizer
from nlp_sum.my_sum.method.extract_summarizer.submodular import SubmodularSummarizer
from nlp_sum.my_sum.method.extract_summarizer.lexrank import LexRankSummarizer
from nlp_sum.my_sum.method.extract_summarizer.lsa import LsaSummarizer
from nlp_sum.my_sum.method.syscom.round_robin import RoundRobinSummarizer
from nlp_sum.test.utils_for_test import get_cn_sentence_length, get_en_sentence_length


class testRoundRobin(unittest.TestCase):

    def test_summarizer(self):
        data_file_path = abspath(dirname(__file__)) + '/data'
        cn_data_file_path = data_file_path + '/chinese/'
        en_data_file_path = data_file_path + '/english/'
        parser_cn = PlaintextParser("chinese")
        parser_en = PlaintextParser("english")

        summarizer_cn = RoundRobinSummarizer("chinese")
        summarizer_cn1 = SubmodularSummarizer("chinese")
        summarizer_cn2 = LexRankSummarizer("chinese")
        summarizer_cn3 = LsaSummarizer("chinese")

        summarizer_en = RoundRobinSummarizer("english")
        summarizer_en1 = SubmodularSummarizer("english")
        summarizer_en2 = LexRankSummarizer("english")
        summarizer_en3 = LsaSummarizer("english")

        document_set_cn = parser_cn.build_documentSet_from_dir(
            cn_data_file_path
        )
        document_set_en = parser_en.build_documentSet_from_dir(
            en_data_file_path
        )

        summary_cn = summarizer_cn(document_set_cn, 100, summarizer_cn1, summarizer_cn2, summarizer_cn3)
        summary_cn_length = sum(get_cn_sentence_length(sentence) for sentence in summary_cn)
        summary_cn_text = ''.join(sentence._texts + '。' for sentence in summary_cn)

        summary_en = summarizer_en(document_set_en, 100, summarizer_en1, summarizer_en2, summarizer_en3)
        summary_en_length = sum(get_en_sentence_length(sentence) for sentence in summary_en)
        summary_en_text = ' '.join(sentence._texts for sentence in summary_en)

        self.assertLessEqual(summary_cn_length, 100)
        self.assertLessEqual(summary_en_length, 100)

        print("--------------------------chinese  round_robin-----------------------------")
        print(summary_cn_text)
        print("the summary length is {}".format(summary_cn_length))
        print("--------------------------english  round_robin-----------------------------")
        print(summary_en_text)
        print("the summary length is {}".format(summary_en_length))
