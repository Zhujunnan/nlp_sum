# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import unittest
import math

from nlp_sum.my_sum.similarity.cosine_sim import compute_tf, compute_idf, cosine_similarity
from nlp_sum.my_sum.models import DocumentSet, Document, Paragraph, Sentence
from .utils_for_test import build_document_en, build_document_cn, build_document_from_string

class TestCosSim(unittest.TestCase):

    def test_compute_tf_idf(self):
        documents = (
            ("this", "is", "a", "example"),
            ("just", "for", "test"),
            ("test", "tf", "and", "idf"),
        )

        tf_metrics = compute_tf(documents)
        idf_metrics = compute_idf(documents)
        expected_tf = [
            {"this": 1/4, "is": 1/4, "a": 1/4, "example": 1/4},
            {"just": 1/3, "for": 1/3, "test": 1/3},
            {"test": 1/4, "tf": 1/4, "and": 1/4, "idf": 1/4},
        ]
        expected_idf = {
            "this": math.log(3/2),
            "is": math.log(3/2),
            "a": math.log(3/2),
            "example": math.log(3/2),
            "just": math.log(3/2),
            "for": math.log(3/2),
            "test": math.log(3/3),
            "tf": math.log(3/2),
            "and": math.log(3/2),
            "idf": math.log(3/2),
        }

        self.assertEqual(tf_metrics, expected_tf)
        self.assertEqual(idf_metrics, expected_idf)

    def test_cosine_similarity_for_the_same_sentence_with_duplicate_words_should_be_one(self):
        """
        We compute similarity of the same sentences. These should be exactly the same and
        therefor have similarity close to 1.0.
        """
        sentence1 = ["this", "sentence", "is", "simple", "sentence"]
        tf1 = {"this": 1/4, "sentence": 1/4, "is": 1/4, "simple": 1/4}
        sentence2 = ["this", "sentence", "is", "simple", "sentence"]
        tf2 = {"this": 1/4, "sentence": 1/4, "is": 1/4, "simple": 1/4}
        idf = {
            "this": 1,
            "sentence": 1,
            "is": 1,
            "simple": 1,
        }

        cosine = cosine_similarity(sentence1, sentence2, tf1, tf2, idf)
        self.assertAlmostEqual(cosine, 1.0)

    def test_cosine_similarity_sentences_with_no_common_word_should_be_zero(self):
        """
        We compute similarity of the sentences without single common word.
        These are considered dissimilar so have similarity close to 0.0.
        """
        sentence1 = ["this", "sentence", "is", "simple", "sentence"]
        tf1 = {"this": 1/2, "sentence": 1.0, "is": 1/2, "simple": 1/2}
        sentence2 = ["that", "paragraph", "has", "some", "words"]
        tf2 = {"that": 1.0, "paragraph": 1.0, "has": 1.0, "some": 1.0, "words": 1.0}
        idf = {
            "this": 2/1,
            "sentence": 2/1,
            "is": 2/1,
            "simple": 2/1,
            "that": 2/1,
            "paragraph": 2/1,
            "has": 2/1,
            "some": 2/1,
            "words": 2/1,
        }

        cosine = cosine_similarity(sentence1, sentence2, tf1, tf2, idf)
        self.assertAlmostEqual(cosine, 0.0)
