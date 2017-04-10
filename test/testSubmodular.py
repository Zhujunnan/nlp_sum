# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import unittest

from os.path import abspath, dirname
from nlp_sum.my_sum.utils import get_stop_words
from nlp_sum.my_sum.parse.plaintext import PlaintextParser
from nlp_sum.my_sum.method.extract_summarizer.submodular import SubmodularSummarizer


class testSubmodular(unittest.TestCase):

    def test_summarizer(self):
        summarizer_cn = SubmodularSummarizer("chinese")
        summarizer_en = SubmodularSummarizer("english")
        summarizer_en_stem = SubmodularSummarizer("english", True)

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
        summarizer_en_stem.stop_words = get_stop_words("english")

        summary_cn = summarizer_cn(document_set_cn, 100)
        summary_cn_len = sum(len(sentence._texts) + 1 for sentence in summary_cn)
        summary_cn_text = ''.join(sentence._texts + '。' for sentence in summary_cn)

        summary_en = summarizer_en(document_set_en, 100)
        summary_en_len = sum(len(sentence.words) for sentence in summary_en)
        summary_en_text = ''.join(sentence._texts for sentence in summary_en)

        summary_en_stem = summarizer_en(document_set_en, 100)
        summary_en_stem_len = sum(len(sentence.words) for sentence in summary_en_stem)
        summary_en_stem_text = ''.join(sentence._texts for sentence in summary_en_stem)

        self.assertLessEqual(summary_cn_len, 100)
        self.assertLessEqual(summary_en_len, 100)
        self.assertLessEqual(summary_en_stem_len, 100)

        print("-----------------------------chinese default---------------------------")
        print(summary_cn_text)
        print("the summary length is {}".format(summary_cn_len))
        print("-----------------------------english default---------------------------")
        print(summary_en_text)
        print("the summary length is {}".format(summary_en_len))
        print("-----------------------------english    stem---------------------------")
        print(summary_en_stem_text)
        print("the summary length is {}".format(summary_en_stem_len))
