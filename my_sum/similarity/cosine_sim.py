# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import math
from collections import Counter


def compute_tf(document_words):
    "or you can use the TfDocumentModel to calculate"
    tf_values = map(Counter, document_words)
    docset_tf  = []
    for document_tf in tf_values:
        metrics = {}
        all_tf = sum(document_tf.values())
        for term, tf in document_tf.items():
            metrics[term] = tf / all_tf
        docset_tf.append(metrics)
    return docset_tf

def compute_idf(document_words):
    idf_metrics = {}
    document_count = len(document_words)
    for document_word in document_words:
        for term in document_word:
            if term not in idf_metrics:
                number_doc = sum(1 for doc in document_words
                                 if term in doc )
                idf_metrics[term] = math.log(
                    document_count / (1 + number_doc)
                )
                if idf_metrics[term] < 0:
                    idf_metrics[term] = 0.0
    return idf_metrics

def cosine_similarity(sentence_words1, sentence_words2, tf1, tf2, idf_metrics):
    """
        We compute idf-modified-cosine(sentence1, sentence2) here.
        It's cosine similarity of these two sentences (vectors) A, B computed as cos(x, y) = A . B / (|A| . |B|)
        Sentences are represented as vector TF*IDF metrics.

        :param sentence_word1:
            tuple or list of words, for example Sentence.words in nlp_sum.my_sum.models.sentence
        :param sentence2:
            tuple or list of words, for example Sentence.words in nlp_sum.my_sum.models.sentence
        :type tf1: dict
        :param tf1:
            Term frequencies of words from document in where 1st sentence is.
        :type tf2: dict
        :param tf2:
            Term frequencies of words from document in where 2nd sentence is.
        :type idf_metrics: dict
        :param idf_metrics:
            Inverted document metrics of the sentences. Every sentence is treated as document for this algorithm.
        :rtype: float
        :return:
            Returns -1.0 for opposite similarity, 1.0 for the same sentence and zero for no similarity between sentences.
        """
    unique_words1 = frozenset(sentence_words1)
    unique_words2 = frozenset(sentence_words2)
    common_words = unique_words1 & unique_words2

    numerator = sum(tf1[term] * tf2[term] * idf_metrics[term]**2
                    for term in common_words)
    denominator1 = sum(
        ((tf1[term] * idf_metrics[term])**2 for term in unique_words1)
    )
    denominator2 = sum(
        ((tf2[term] * idf_metrics[term])**2 for term in unique_words2)
    )

    if denominator1 > 0 and denominator2 > 0:
        return numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))
    else:
        return 0.0
