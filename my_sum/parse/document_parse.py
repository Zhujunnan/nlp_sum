# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from nlp_sum.my_sum.nlp.Tokenizer import Tokenizer


class DocumentParser(object):

    def __init__(self, language):
        self._tokenizer = Tokenizer(language)
        self.language = language

    def tokenize_sentences(self, text):
        return self._tokenizer.to_sentences(text)

    def tokenize_words(self, sentence):
        return self._tokenizer.to_words(sentence)
