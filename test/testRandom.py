# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import unittest

from os.path import abspath, dirname
from nlp_sum.my_sum.parse.plaintext import PlaintextParser
from nlp_sum.my_sum.method.extract_summarizer.random import RandomSummarizer


class testRandom(unittest.TestCase):

    def test_summarizer(self):
        summarizer_cn = RandomSummarizer("chinese")
        summarizer_en = RandomSummarizer("english")

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

        summary_cn = summarizer_cn(document_set_cn, 100)
        summary_cn_length = sum(len(sentence._texts) + 1 for sentence in summary_cn)
        summary_cn_text = ''.join(sentence._texts + '。' for sentence in summary_cn)

        summary_en = summarizer_en(document_set_en, 100)
        summary_en_length = sum(len(sentence.words) for sentence in summary_en)
        summary_en_text = ''.join(sentence._texts for sentence in summary_en)

        self.assertLessEqual(summary_cn_length, 100)
        self.assertLessEqual(summary_en_length, 100)

        print("----------------------------chinese random--------------------------------")
        print(summary_cn_text)
        print("the summary length is {}".format(summary_cn_length))

        print("----------------------------english random--------------------------------")
        print(summary_en_text)
        print("the summary length is {}".format(summary_en_length))
