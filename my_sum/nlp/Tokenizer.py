# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import sys
import re
import nltk
import jieba

#from os.path import abspath

#sys.path.append(abspath(".."))
#from util import to_unicode
from ..utils import to_unicode


def chinese_word_segment(unicode_str):
    """jieba cut result is a generator"""
    return list(jieba.cut(unicode_str))

def chinese_sentence_segment(text):
    """
    segment text into sentence, but remove the delimiters
    remember to add delimiter after sentences in final summary
    """
    sentence_delimiters = ['?', '!', ';', '？', '！',
                           '。', '；', '……', '…', '\n']
    delimiters = set([to_unicode(item) for item in sentence_delimiters])
    result = [to_unicode(text)]

    for sep in delimiters:
        text, result = result, []
        for seq in text:
            result += seq.split(sep)
            res = [sent.strip() for sent in result if len(sent.strip()) > 0]

    return res

def english_sentence_segment(text):
    """segment text into sentence"""
    try:
        sent_detector = nltk.data.load(
            'tokenizers/punkt/english.pickle'
        )

        extra_abbrev = ["e.g", "al", "i.e"]
        sent_detector._params.abbrev_types.update(extra_abbrev)
        return sent_detector.tokenize(text)
    except LookupError as e:
        raise LookupError(
            "NLTK tokenizers are missing. Download them by following command: "
            '''python -c "import nltk; nltk.download('punkt')"'''
        )


class Tokenizer(object):
    """Language dependent tokenizer of text document"""
    LANGUAGE_TOKENIZER = {
        "english" : nltk.word_tokenize,
        "chinese" : chinese_word_segment,
    }

    SENTENCE_SEGMENT = {
        "english" : english_sentence_segment,
        "chinese" : chinese_sentence_segment,
    }

    WORD_PATTERN = re.compile(r"^[^\W\d_]+$", re.UNICODE)


    def __init__(self, language):
        self._language = language.lower()
        self.tokenizer = self._get_sentence_tokenizer(language)
        self.sent_segment = self._get_sentence_segment(language)

    @property
    def language(self):
        return self._language

    def _get_sentence_tokenizer(self, language):
        try:
            return self.LANGUAGE_TOKENIZER[language]
        except KeyError as e:
            raise LookupError("The language is not supported here!")

    def _get_sentence_segment(self, language):
        try:
            return self.SENTENCE_SEGMENT[language]
        except KeyError as e:
            raise LookupError("The language is not supported here!")

    def to_words(self, sentence):
        words = self.tokenizer(to_unicode(sentence))
        return tuple(filter(self.is_word, words))

    def to_sentences(self, text):
        sentences = self.sent_segment(to_unicode(text))
        return tuple(map(unicode.strip, sentences))

    @staticmethod
    def is_word(word):
        """to filter non-word"""
        return bool(Tokenizer.WORD_PATTERN.search(word))

