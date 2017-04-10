# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import unittest

from os.path import abspath, dirname
from nlp_sum.my_sum.utils import get_stop_words
from nlp_sum.my_sum.parse.plaintext import PlaintextParser
from nlp_sum.my_sum.method.extract_summarizer.conceptILP import conceptILPSummarizer


class testILP(unittest.TestCase):

    def test_summarizer(self):
        summarizer_en = conceptILPSummarizer("english")
        summarizer_en_stem = conceptILPSummarizer("english", True)
        summarizer_cn = conceptILPSummarizer("chinese")

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

        summary_cn_ilp = summarizer_cn(document_set_cn, 100, method="ilp")
        summary_cn_ilp_len = sum(len(sentence._texts) + 1 for sentence in summary_cn_ilp)
        summary_cn_ilp_text = ''.join(sentence._texts + '。' for sentence in summary_cn_ilp)

        summary_cn_greedy = summarizer_cn(document_set_cn, 100, method="greedy")
        summary_cn_greedy_len = sum(len(sentence._texts) + 1 for sentence in summary_cn_greedy)
        summary_cn_greedy_text = ''.join(sentence._texts + '。' for sentence in summary_cn_greedy)

        summary_cn_tabu = summarizer_cn(document_set_cn, 100, method="tabu")
        summary_cn_tabu_len = sum(len(sentence._texts) + 1 for sentence in summary_cn_tabu)
        summary_cn_tabu_text = ''.join(sentence._texts + '。' for sentence in summary_cn_tabu)

        summary_en_ilp = summarizer_en(document_set_en, 100, method="ilp")
        summary_en_ilp_len = sum(len(sentence.words) for sentence in summary_en_ilp)
        summary_en_ilp_text = ''.join(sentence._texts for sentence in summary_en_ilp)

        summary_en_greedy = summarizer_en(document_set_en, 100, method="greedy")
        summary_en_greedy_len = sum(len(sentence.words) for sentence in summary_en_greedy)
        summary_en_greedy_text = ''.join(sentence._texts for sentence in summary_en_greedy)

        summary_en_tabu = summarizer_en(document_set_en, 100, method="tabu")
        summary_en_tabu_len = sum(len(sentence.words) for sentence in summary_en_tabu)
        summary_en_tabu_text = ''.join(sentence._texts for sentence in summary_en_tabu)

        summary_en_stem_ilp = summarizer_en_stem(document_set_en, 100, method="ilp")
        summary_en_stem_ilp_len = sum(len(sentence.words) for sentence in summary_en_stem_ilp)
        summary_en_stem_ilp_text = ''.join(sentence._texts for sentence in summary_en_stem_ilp)

        summary_en_stem_greedy = summarizer_en_stem(document_set_en, 100, method="greedy")
        summary_en_stem_greedy_len = sum(len(sentence.words) for sentence in summary_en_stem_greedy)
        summary_en_stem_greedy_text = ''.join(sentence._texts for sentence in summary_en_stem_greedy)

        summary_en_stem_tabu = summarizer_en_stem(document_set_en, 100, method="tabu")
        summary_en_stem_tabu_len = sum(len(sentence.words) for sentence in summary_en_stem_tabu)
        summary_en_stem_tabu_text = ''.join(sentence._texts for sentence in summary_en_stem_tabu)


        self.assertLessEqual(summary_cn_ilp_len, 100)
        self.assertLessEqual(summary_cn_greedy_len, 100)
        self.assertLessEqual(summary_cn_tabu_len, 100)

        self.assertLessEqual(summary_en_ilp_len, 100)
        self.assertLessEqual(summary_en_greedy_len, 100)
        self.assertLessEqual(summary_en_tabu_len, 100)

        self.assertLessEqual(summary_en_stem_ilp_len, 100)
        self.assertLessEqual(summary_en_stem_greedy_len, 100)
        self.assertLessEqual(summary_en_stem_tabu_len, 100)

        print("--------------------------chinese      ILP-----------------------------")
        print(summary_cn_ilp_text)
        print("the summary length is {}".format(summary_cn_ilp_len))
        print("--------------------------chinese   greedy-----------------------------")
        print(summary_cn_greedy_text)
        print("the summary length is {}".format(summary_cn_greedy_len))
        print("--------------------------chinese     tabu-----------------------------")
        print(summary_cn_tabu_text)
        print("the summary length is {}".format(summary_cn_tabu_len))
        print("--------------------------english      ILP-----------------------------")
        print(summary_en_ilp_text)
        print("the summary length is {}".format(summary_en_ilp_len))
        print("--------------------------english   greedy-----------------------------")
        print(summary_en_greedy_text)
        print("the summary length is {}".format(summary_en_greedy_len))
        print("--------------------------english     tabu-----------------------------")
        print(summary_en_tabu_text)
        print("the summary length is {}".format(summary_en_tabu_len))
        print("--------------------------english Stem ILP-----------------------------")
        print(summary_en_stem_ilp_text)
        print("the summary length is {}".format(summary_en_stem_ilp_len))
        print("--------------------------eng   StemGreedy-----------------------------")
        print(summary_en_stem_greedy_text)
        print("the summary length is {}".format(summary_en_stem_greedy_len))
        print("--------------------------english StemTabu-----------------------------")
        print(summary_en_stem_tabu_text)
        print("the summary length is {}".format(summary_en_stem_tabu_len))
