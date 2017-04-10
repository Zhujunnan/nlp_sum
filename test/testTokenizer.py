# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import unittest
#import sys
#from os.path import abspath

#sys.path.append(abspath(".."))

from nlp_sum.my_sum.nlp.Tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):

    def test_missing_language(self):
        self.assertRaises(LookupError, Tokenizer, "mandarin")

    def test_ensure_tokenizer_available(self):
        tokenizer_en = Tokenizer("english")
        self.assertEqual("english", tokenizer_en.language)
        tokenizer_cn = Tokenizer("chinese")
        self.assertEqual("chinese", tokenizer_cn.language)

        sentence_en = "You are very beautiful."
        sentence_cn = "我来到北京清华大学"

        expected_en_words = (
            "You", "are",
            "very", "beautiful",
        )

        expected_cn_words = (
            "我", "来到",
            "北京", "清华大学",
        )

        self.assertEqual(expected_en_words, tokenizer_en.to_words(sentence_en))
        self.assertEqual(expected_cn_words, tokenizer_cn.to_words(sentence_cn))

    def test_ensure_segment_available(self):
        tokenizer_en = Tokenizer("english")
        tokenizer_cn = Tokenizer("chinese")

        sentence_en = "There are two sentences here. Am I right?"
        sentence_cn = "这里貌似有两句话。不知道我说的对不对。"

        self.assertEqual(len(tokenizer_en.to_sentences(sentence_en)), 2)
        self.assertEqual(len(tokenizer_cn.to_sentences(sentence_cn)), 2)
