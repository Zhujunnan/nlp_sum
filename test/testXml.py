# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import unittest
from os.path import abspath, dirname

from nlp_sum.my_sum.utils import to_unicode
from nlp_sum.my_sum.parse.xml_parse import XmlParser


class testXml(unittest.TestCase):

    def test_parse_file(self):
        data_file_path = abspath(dirname(__file__)) + '/data'
        xml_file_path = data_file_path + '/xml/'
        xml_file = xml_file_path + 'xml_one'

        parser_en = XmlParser("english")

        document_set_en = parser_en.build_document_from_dir(
            xml_file_path
        )

        document_en = parser_en.build_document_from_file(
            xml_file
        )

        self.assertEqual(len(document_en.paragraphs), 1)
        self.assertEqual(len(document_en.sentences), 5)
        self.assertEqual(len(document_set_en.documents), 2)
        self.assertEqual(len(document_set_en.paragraphs), 2)
        self.assertEqual(len(document_set_en.sentences), 12)
