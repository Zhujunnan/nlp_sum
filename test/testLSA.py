# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import unittest

from os.path import abspath, dirname
from nlp_sum.my_sum.utils import get_stop_words
from nlp_sum.my_sum.parse.plaintext import PlaintextParser
from nlp_sum.my_sum.method.extract_summarizer.lsa import LsaSummarizer

from nlp_sum.my_sum.similarity.cosine_sim import compute_tf, compute_idf
from nlp_sum.my_sum.similarity.cosine_sim import cosine_similarity

from nlp_sum.test.utils_for_test import get_cn_sentence_length, get_en_sentence_length


class testLSA(unittest.TestCase):

    def test_summarizer(self):
        summarizer_en = LsaSummarizer("english")
        summarizer_en_stem = LsaSummarizer("english", True)
        summarizer_cn = LsaSummarizer("chinese")

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

        summary_cn = summarizer_cn(document_set_cn, 100)
        summary_cn_length = sum(get_cn_sentence_length(sentence) for sentence in summary_cn)
        summary_cn_text = ''.join(sentence._texts + '。' for sentence in summary_cn)

        # summary_cn_mmr = summarizer_cn(document_set_cn, 100, method="MMR")
        summary_cn_mmr = summarizer_cn(document_set_cn, 100, method="MMR", metric="tfidf")
        summary_cn_mmr_length = sum(get_cn_sentence_length(sentence) for sentence in summary_cn_mmr)
        summary_cn_text_mmr = ''.join(sentence._texts + '。' for sentence in summary_cn_mmr)

        summary_en_tfidf = summarizer_en(document_set_en, 100, method="MMR", metric="tfidf")
        summary_en_tfidf_length = sum(get_en_sentence_length(sentence) for sentence in summary_en_tfidf)
        summary_en_text_tfidf = ' '.join(sentence._texts for sentence in summary_en_tfidf)

        summary_en_mmr = summarizer_en(document_set_en, 100, method="MMR")
        summary_en_mmr_length = sum(get_en_sentence_length(sentence) for sentence in summary_en_mmr)
        summary_en_text_mmr = ' '.join(sentence._texts for sentence in summary_en_mmr)

        print("-----------------------------chinese default-------------------------------")
        print(summary_cn_text)
        print("the summary length is {}".format(summary_cn_length))
        print("-----------------------------chinese     MMR-------------------------------")
        print(summary_cn_text_mmr)
        print("the summary length is {}".format(summary_cn_mmr_length))
        print("-----------------------------english   tfidf-------------------------------")
        print(summary_en_text_tfidf)
        print("the summary length is {}".format(summary_en_tfidf_length))
        print("-----------------------------english     MMR-------------------------------")
        print(summary_en_text_mmr)
        print("the summary length is {}".format(summary_en_mmr_length))

        self.assertLessEqual(summary_en_tfidf_length, 100)
        self.assertLessEqual(summary_en_mmr_length, 100)
        self.assertLessEqual(summary_cn_length, 100)
        self.assertLessEqual(summary_cn_mmr_length, 100)

        #diction = summarizer_cn._create_dictionary(document_set_cn)
        #print(len(diction))
        #sentence_number = [len(doc.sentences) for doc in document_set_cn.documents]
        #print(sentence_number)
