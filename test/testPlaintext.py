# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import unittest
from os.path import abspath, dirname

from nlp_sum.my_sum.utils import to_unicode
from nlp_sum.my_sum.parse.plaintext import PlaintextParser


class testPlaintext(unittest.TestCase):

    def test_parse_file(self):
        data_file_path = abspath(dirname(__file__)) + '/data'

        cn_data_file_path = data_file_path + '/chinese/'
        en_data_file_path = data_file_path + '/english/'
        cn_file = cn_data_file_path + 'chinese_one.txt'
        en_file = en_data_file_path + 'english_one.txt'

        parser_cn = PlaintextParser("chinese")
        parser_en = PlaintextParser("english")
        document_cn = parser_cn.build_document_from_file(
            cn_file
        )
        document_en = parser_en.build_document_from_file(
            en_file
        )

        document_set_cn = parser_cn.build_documentSet_from_dir(
            cn_data_file_path
        )
        document_set_en = parser_en.build_documentSet_from_dir(
            en_data_file_path
        )

        self.assertEqual(len(document_set_cn.documents), 2)
        self.assertEqual(len(document_set_cn.paragraphs), 10)
        self.assertEqual(len(document_set_en.documents), 2)
        self.assertEqual(len(document_set_en.paragraphs), 6)
        self.assertEqual(len(document_cn.paragraphs), 5)
        self.assertEqual(len(document_cn.sentences), 7)
        self.assertEqual(len(document_en.paragraphs), 4)
        self.assertEqual(len(document_en.sentences), 13)
