# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import unittest

from nlp_sum.my_sum.nlp.Tokenizer import Tokenizer
from nlp_sum.my_sum.models.TF import TfDocumentModel

class testTfDocumentModel(unittest.TestCase):
    def test_no_tokenizer_with_string(self):
        self.assertRaises(ValueError, TfDocumentModel, "string without tokenizer")

    def test_pretokenized_words_frequency(self):
        cn_model = TfDocumentModel(("中文", "信息", "处理", "中文", "中文", "处理"))
        en_model = TfDocumentModel(("Natural", "languge", "processing"))

        self.assertEqual(cn_model.term_frequency("中文"), 3)
        self.assertEqual(en_model.term_frequency("natural"), 1)
        self.assertEqual(cn_model.term_frequency("中国"), 0)
        self.assertEqual(en_model.term_frequency("eng"), 0)
        self.assertEqual(cn_model.most_frequent_terms(), ("中文", "处理", "信息"))
        self.assertEqual(en_model.most_frequent_terms(), ("processing", "natural", "languge"))

    def test_magnitude(self):
        tokenizer_en = Tokenizer("english")
        tokenizer_cn = Tokenizer("chinese")
        text_en = "i am very happy"
        text_cn = "我来到北京清华大学"
        model_en = TfDocumentModel(text_en, tokenizer_en)
        model_cn = TfDocumentModel(text_cn, tokenizer_cn)

        self.assertAlmostEqual(model_en.magnitude, 2.0)
        self.assertAlmostEqual(model_cn.magnitude, 2.0)

    def test_terms(self):
        tokenizer = Tokenizer("english")
        text = "wA wB wC wD wB wD wE"
        model = TfDocumentModel(text, tokenizer)

        terms = tuple(sorted(model.terms))
        self.assertEqual(terms, ("wa", "wb", "wc", "wd", "we"))

    def test_term_frequency(self):
        tokenizer = Tokenizer("english")
        text = "wA wB wC wA wA wC wD wCwB"
        model = TfDocumentModel(text, tokenizer)

        self.assertEqual(model.term_frequency("wa"), 3)
        self.assertEqual(model.term_frequency("wb"), 1)
        self.assertEqual(model.term_frequency("wc"), 2)
        self.assertEqual(model.term_frequency("wd"), 1)
        self.assertEqual(model.term_frequency("wcwb"), 1)
        self.assertEqual(model.term_frequency("we"), 0)
        self.assertEqual(model.term_frequency("missing"), 0)
        self.assertAlmostEqual(model.term_frequency_normalized("missing"), 0)
        self.assertAlmostEqual(model.term_frequency_normalized("wa", "max"), 1.0)
        self.assertAlmostEqual(model.term_frequency_normalized("wa"), 3/8)

    def test_most_frequent_terms(self):
        tokenizer = Tokenizer("english")
        text = "wE wD wC wB wA wE WD wC wB wE wD WE wC wD wE"
        model = TfDocumentModel(text, tokenizer)

        self.assertEqual(model.most_frequent_terms(1), ("we",))
        self.assertEqual(model.most_frequent_terms(2), ("we", "wd"))
        self.assertEqual(model.most_frequent_terms(3), ("we", "wd", "wc"))
        self.assertEqual(model.most_frequent_terms(4), ("we", "wd", "wc", "wb"))
        self.assertEqual(model.most_frequent_terms(5), ("we", "wd", "wc", "wb", "wa"))
        self.assertEqual(model.most_frequent_terms(), ("we", "wd", "wc", "wb", "wa"))

    def test_most_frequent_terms_empty(self):
        tokenizer = Tokenizer("english")
        model = TfDocumentModel("", tokenizer)

        self.assertEqual(model.most_frequent_terms(), ())
        self.assertEqual(model.most_frequent_terms(10), ())

    def test_most_frequent_terms_negative_count(self):
        tokenizer = Tokenizer("english")
        model = TfDocumentModel("text", tokenizer)

        self.assertRaises(ValueError, model.most_frequent_terms, -1)

    def test_normalized_words_frequencies(self):
        words = "a b c d e c b d c e e d e d e".split()
        model = TfDocumentModel(tuple(words))

        self.assertAlmostEqual(model.normalized_term_frequency("a"), 1/5)
        self.assertAlmostEqual(model.normalized_term_frequency("b"), 2/5)
        self.assertAlmostEqual(model.normalized_term_frequency("c"), 3/5)
        self.assertAlmostEqual(model.normalized_term_frequency("d"), 4/5)
        self.assertAlmostEqual(model.normalized_term_frequency("e"), 5/5)
        self.assertAlmostEqual(model.normalized_term_frequency("z"), 0.0)

        self.assertEqual(model.most_frequent_terms(), ("e", "d", "c", "b", "a"))

    def test_normalized_words_frequencies_with_smoothing_term(self):
        words = "a b c d e c b d c e e d e d e".split()
        model = TfDocumentModel(tuple(words))

        self.assertAlmostEqual(model.normalized_term_frequency("a", 0.5), 0.5 + 1/10)
        self.assertAlmostEqual(model.normalized_term_frequency("b", 0.5), 0.5 + 2/10)
        self.assertAlmostEqual(model.normalized_term_frequency("c", 0.5), 0.5 + 3/10)
        self.assertAlmostEqual(model.normalized_term_frequency("d", 0.5), 0.5 + 4/10)
        self.assertAlmostEqual(model.normalized_term_frequency("e", 0.5), 0.5 + 5/10)
        self.assertAlmostEqual(model.normalized_term_frequency("z", 0.5), 0.5)

        self.assertEqual(model.most_frequent_terms(), ("e", "d", "c", "b", "a"))
