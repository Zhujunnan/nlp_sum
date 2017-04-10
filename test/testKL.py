# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, unicode_literals

import unittest

from os.path import abspath, dirname
from nlp_sum.my_sum.utils import get_stop_words
from nlp_sum.my_sum.parse.plaintext import PlaintextParser
from nlp_sum.my_sum.method.extract_summarizer.kl import KLSummarizer
from nlp_sum.test.utils_for_test import get_cn_sentence_length, get_en_sentence_length


class testKL(unittest.TestCase):

    def test_stopwords(self):
        stop_words_cn = get_stop_words("chinese")
        stop_words_en = get_stop_words("english")

        self.assertIn("啊", stop_words_cn)
        self.assertIn("and", stop_words_en)

    def test_summarizer(self):
        summarizer_en = KLSummarizer("english")
        summarizer_en_stem = KLSummarizer("english", True)
        summarizer_cn = KLSummarizer("chinese")
        summarizer_cn.stop_words = get_stop_words("chinese")

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
        summary_cn_length = sum(get_cn_sentence_length(sentence) for sentence in summary_cn)
        summary_cn_text = ''.join(sentence._texts + '。' for sentence in summary_cn)

        summary_en = summarizer_en(document_set_en, 100)
        summary_en_length = sum(get_en_sentence_length(sentence) for sentence in summary_en)
        summary_en_text = ' '.join(sentence._texts for sentence in summary_en)

        summary_en_stem = summarizer_en_stem(document_set_en, 100)
        summary_en_stem_length = sum(get_en_sentence_length(sentence) for sentence in summary_en_stem)
        summary_en_stem_text = ' '.join(sentence._texts for sentence in summary_en_stem)

        self.assertLessEqual(summary_cn_length, 100)
        self.assertLessEqual(summary_en_length, 100)
        self.assertLessEqual(summary_en_stem_length, 100)

        print("--------------------------chinese   KL-----------------------------")
        print(summary_cn_text)
        print("the summary length is {}".format(summary_cn_length))
        print("--------------------------english   KL-----------------------------")
        print(summary_en_text)
        print("the summary length is {}".format(summary_en_length))
        print("--------------------------english stem-----------------------------")
        print(summary_en_stem_text)
        print("the summary length is {}".format(summary_en_stem_length))

        # print(summary_cn_text)
        # print(summary_en_text)
        # print(summary_en_stem)
        # print summarizer_en_stem._get_content_words_in_sentence(summary_en[0])
        # print summarizer_en._get_content_words_in_sentence(summary_en[0])
        word_list = summarizer_en_stem._get_content_words_in_sentence(summary_en[0])
        word_stem_list = summarizer_en._get_content_words_in_sentence(summary_en[0])
        self.assertNotEqual(word_list, word_stem_list)

    def test_compute_word_freq(self):
        summarizer_en = KLSummarizer("english")
        words = ["one", "two", "three", "four", "one"]
        freq = summarizer_en._compute_word_freq(words)

        self.assertEqual(freq.get("one", 0), 2)
        self.assertEqual(freq.get("two", 0), 1)

    def test_joint_freq(self):
        summarizer_en = KLSummarizer("english")
        word1 = ["one", "two", "three", "four"]
        word2 = ["one", "two", "three", "five"]
        freq = summarizer_en._joint_freq(word1, word2)

        self.assertAlmostEqual(freq['one'], 2/8)
        self.assertAlmostEqual(freq['five'], 1/8)

    def test_kl_divergence(self):
        summarizer_en = KLSummarizer("english")
        word1 = {"one": 0.35, "two": 0.5, "three": 0.15}
        word2 = {"one": 1.0/3.0, "two": 1.0/3.0, "three": 1.0/3.0}

        # This value comes from scipy.stats.entropy(w2_, w1_)
        # Note: the order of params is different
        kl_correct = 0.11475080798005841
        self.assertAlmostEqual(summarizer_en._kl_divergence(word1, word2), kl_correct)

        word1 = {"one": 0.1, "two": 0.2, "three": 0.7}
        word2 = {"one": 0.2, "two": 0.4, "three": 0.4}

        # This value comes from scipy.stats.entropy(w2_, w1_)
        # Note: the order of params is different
        kl_correct = 0.1920419931617981
        self.assertAlmostEqual(summarizer_en._kl_divergence(word1, word2), kl_correct)

        summary_frequences = {"one": 0.35, "two": 0.5, "three": 0.15, "four": 0.9}
        document_frequencies = {"one": 1.0/3.0, "two": 1.0/3.0, "three": 1.0/3.0}

        # This value comes from scipy.stats.entropy(w2_, w1_)
        # Note: the order of params is different
        kl_correct = 0.11475080798005841
        self.assertAlmostEqual(
            summarizer_en._kl_divergence(summary_frequences, document_frequencies),
            kl_correct
        )
