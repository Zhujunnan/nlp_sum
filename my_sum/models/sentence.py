# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

# from functools import total_ordering
from ..utils import cached_property, to_unicode, to_string, unicode_compatible


# @total_ordering
@unicode_compatible
class Sentence(object):
    # __slots__ = (
    #     "_texts",
    #     "_cached_property_words",
    #     "_tokenizer",
    # )

    def __init__(self, text, tokenizer):
        self._texts = to_unicode(text).strip()
        self._tokenizer = tokenizer

    @cached_property
    def words(self):
        return self._tokenizer.to_words(self._texts)

    def __eq__(self, sentence):
        assert isinstance(sentence, Sentence), "sentence must be Sentence"
        return self._texts == sentence._texts

    def __ne__(self, sentence):
        return not self.__eq__(sentence)

    def __unicode__(self):
        return self._texts

    def __repr__(self):
        return to_string("{0}").format(self.__str__())
