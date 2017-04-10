# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import unittest

from nlp_sum.my_sum.nlp.Tokenizer import Tokenizer
from nlp_sum.my_sum.utils import to_unicode
from nlp_sum.my_sum.models import DocumentSet, Document, Paragraph, Sentence
from .utils_for_test import build_document_en, build_document_cn, build_document_from_string

class TestDoc(unittest.TestCase):

    def  test_unique_words(self):
        document_en = build_document_en(
            ("it is a beautiful day today", "I want to go for a walk",),
            ("I really like it",),
        )

        returned_en = tuple(sorted(frozenset(document_en.words)))
        expected_en = (
            "I", "a", "beautiful", "day", "for", "go", "is", "it",
            "like", "really", "to", "today", "walk", "want",
        )

        document_cn = build_document_cn(
            ("我爱这片土地",),
            ("因为这是我的故乡",),
        )
        returned_cn = tuple(sorted(frozenset(document_cn.words)))
        expected_cn = (
            "因为", "土地", "我", "故乡",
            "是", "爱", "的", "这", "这片",
        )

        self.assertEqual(expected_en, returned_en)
        self.assertEqual(expected_cn, returned_cn)

    def test_sentences(self):
        document_en = build_document_from_string("""
            it is a beautiful day today
            I want to go for a walk
            I really like it
        """)

        document_cn = build_document_from_string("""
            我爱这片土地
            因为这是我的故乡
        """, language="chinese")

        self.assertEqual(len(document_en.sentences), 3)
        self.assertEqual(len(document_cn.sentences), 2)
        self.assertEqual(unicode(document_en.sentences[0]),
                         "it is a beautiful day today")
        self.assertEqual(unicode(document_en.sentences[1]),
                         "I want to go for a walk")
        self.assertEqual(unicode(document_en.sentences[2]),
                         "I really like it")

    def test_Docset(self):
        document_one = build_document_from_string("""
            it is a beautiful day today
            I want to go for a walk
            I really like it
        """)

        document_two = build_document_from_string("""
            it is a beautiful day today
            I want to go for a walk
            I really like it. it is amazing.
        """)

        document_set = DocumentSet([document_one, document_two])
        self.assertEqual(len(document_set.documents), 2)
        self.assertEqual(len(document_set.paragraphs), 6)
        self.assertEqual(len(document_set.sentences), 7)
        self.assertEqual(len(document_set.words), 37)

    def test_sentence_equal(self):
        sentence_one = Sentence("", Tokenizer("english"))
        sentence_two = Sentence("", Tokenizer("english"))
        self.assertEqual(sentence_one, sentence_two)

        sentence_one = Sentence("another example", Tokenizer("english"))
        sentence_two = Sentence("another example", Tokenizer("english"))
        self.assertEqual(sentence_one, sentence_two)

        sentence_one = Sentence("example", Tokenizer("english"))
        sentence_two = Sentence("another", Tokenizer("english"))
        self.assertNotEqual(sentence_one, sentence_two)
