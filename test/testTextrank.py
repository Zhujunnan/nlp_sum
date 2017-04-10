# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import unittest

from os.path import abspath, dirname
from nlp_sum.my_sum.utils import get_stop_words
from nlp_sum.my_sum.parse.plaintext import PlaintextParser
from nlp_sum.my_sum.method.extract_summarizer.textrank import TextRankSummarizer


class testTextrank(unittest.TestCase):

    def test_summarizer(self):
        summarizer_cn = TextRankSummarizer("chinese")
        summarizer_en = TextRankSummarizer("english")

        data_file_path = abspath(dirname(__file__)) + '/data'
        cn_data_file_path = data_file_path + '/chinese/'
        en_data_file_path = data_file_path + '/english/'
        parser_cn = PlaintextParser("chinese")
        parser_en = PlaintextParser("english")

        document_set_cn = parser_cn.build_documentSet_from_dir(
            cn_data_file_path
        )
        document_set_en = parser_en.build_documentSet_from_dir(
            en_data_file_path
        )

        summarizer_cn.stop_words = get_stop_words("chinese")
        summarizer_en.stop_words = get_stop_words("english")

        summary_cn = summarizer_cn(document_set_cn, 100)
        summary_cn_len = sum(len(sentence._texts) + 1 for sentence in summary_cn)
        summary_cn_text = ''.join(sentence._texts + '。' for sentence in summary_cn)

        summary_cn_mmr = summarizer_cn(document_set_cn, 100, method="MMR")
        summary_cn_mmr_len = sum(len(sentence._texts) + 1 for sentence in summary_cn_mmr)
        summary_cn_mmr_text = ''.join(sentence._texts + '。' for sentence in summary_cn_mmr)

        summary_en = summarizer_en(document_set_en, 100)
        summary_en_len = sum(len(sentence.words) for sentence in summary_en)
        summary_en_text = ''.join(sentence._texts for sentence in summary_en)

        summary_en_mmr = summarizer_en(document_set_en, 100, method="MMR")
        summary_en_mmr_len = sum(len(sentence.words) for sentence in summary_en_mmr)
        summary_en_mmr_text = ''.join(sentence._texts for sentence in summary_en_mmr)

        self.assertLessEqual(summary_cn_len, 100)
        self.assertLessEqual(summary_cn_mmr_len, 100)
        self.assertLessEqual(summary_en_len, 100)
        self.assertLessEqual(summary_en_mmr_len, 100)

        print("-----------------------------chinese default---------------------------")
        print(summary_cn_text)
        print("the summary length is {}".format(summary_cn_len))
        print("-----------------------------chinese     MMR---------------------------")
        print(summary_cn_mmr_text)
        print("the summary length is {}".format(summary_cn_mmr_len))
        print("-----------------------------english default---------------------------")
        print(summary_en_text)
        print("the summary length is {}".format(summary_en_len))
        print("-----------------------------english     MMR---------------------------")
        print(summary_en_mmr_text)
        print("the summary length is {}".format(summary_en_mmr_len))
