# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import os

from ..utils import to_unicode, cached_property
from ..models import Sentence, Paragraph, Document, DocumentSet
from .document_parse import DocumentParser


class PlaintextParser(DocumentParser):

    def __init(self, language):
        super(PlaintextParser, self).__init__(language)

    def build_documentSet_from_dir(self, path):
        if not path.endswith("/"):
            path = path + "/"
        file_list = os.listdir(path)
        # filter file which is hidden file (like .DS_Store)
        file_list = filter(lambda file: not file.startswith("."), file_list)
        file_path_list = [path + file_name for file_name in file_list]
        Document_list = [
            self.build_document_from_file(self._tokenizer, file_path)
            for file_path in file_path_list
        ]
        return DocumentSet(Document_list)

    @staticmethod
    def build_document_from_file(tokenizer, file_path):
        # build Document from a single file
        try:
            with open(file_path, 'r') as file:
                text = to_unicode(file.read())
        except IOError:
            print("please check if {} is valid".format(file_path))

        sentences = []
        paragraphs =[]
        for line in text.strip().splitlines():
            line = line.lstrip()
            if line:
                sentence_tuple = tokenizer.to_sentences(line)
                for sentence in sentence_tuple:
                    sentences.append(Sentence(sentence, tokenizer))
                paragraphs.append(Paragraph(sentences))
                sentences = []
            else:
                continue
        return Document(paragraphs)
