# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, unicode_literals, print_function

from os.path import abspath, dirname
from nlp_sum.my_sum.parse.plaintext import PlaintextParser
from nlp_sum.my_sum.method.extract_summarizer.submodular import SubmodularSummarizer
from nlp_sum.my_sum.method.extract_summarizer.lexrank import LexRankSummarizer
from nlp_sum.my_sum.method.extract_summarizer.lsa import LsaSummarizer

from nlp_sum.my_sum.similarity.cosine_sim import compute_tf, compute_idf
from nlp_sum.my_sum.similarity.cosine_sim import cosine_similarity
from nlp_sum.my_sum.models import Document, Paragraph

from nlp_sum.test.utils_for_test import get_cn_sentence_length, get_en_sentence_length


# summarizer_en1 = SubmodularSummarizer("english")
# summarizer_en2 = LexRankSummarizer("english")
# summarizer_en3 = LsaSummarizer("english")

class GraphSummarizer(object):

    def __init__(self, language="english"):
        self.language = language.lower()

        if self.language.startswith("en"):
            self._get_sentence_length = get_en_sentence_length
        if self.language.startswith("ch"):
            self._get_sentence_length = get_cn_sentence_length

    def __call__(self, documentSet, words_limit, summarizer1, summarizer2, summarizer3):
        return self.graph(documentSet, words_limit, summarizer1, summarizer2, summarizer3)

    def graph(self, document, words_limit, summarizer1, summarizer2, summarizer3):
        summary_1 = summarizer1(document, words_limit, summary_order="rating")
        summary_2 = summarizer2(document, words_limit, summary_order="rating")
        summary_3 = summarizer3(document, words_limit, summary_order="rating")
        summary_2 = filter(lambda x: x not in summary_1, summary_2)
        summary_3 = filter(lambda x: (x not in summary_1) and (x not in summary_2), summary_3)
        summarizer = LexRankSummarizer(self.language)

        paragraphs = [Paragraph(summary_1), Paragraph(summary_2), Paragraph(summary_3)]
        document_sum = Document(paragraphs)
        sentences = document.sentences

        graph_summary = summarizer(document_sum, words_limit)
        graph_summary = sorted(graph_summary, key=lambda x: sentences.index(x))

        return graph_summary
