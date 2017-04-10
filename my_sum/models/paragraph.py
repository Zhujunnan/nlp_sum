# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from itertools import chain
from ..utils import to_unicode, unicode_compatible, cached_property
from .sentence import Sentence


@unicode_compatible
class Paragraph(object):
    # __slots__ = (
    #     "_sentences",
    #     "_cached_property_sentences",
    #     "_cached_property_words",
    # )

    def __init__(self, sentences):
        """sentences shoule be a list or tuple"""
        sentences = tuple(sentences)
        for sentence in sentences:
            if not isinstance(sentence, Sentence):
                raise TypeError("Only instances of class 'Sentence' are allowed.")

        self._sentences = sentences

    @cached_property
    def sentences(self):
        return tuple(sentence for sentence in self._sentences)

    @cached_property
    def words(self):
        return tuple(chain(
            *(sentence.words for sentence in self._sentences)
        ))

    def __unicode__(self):
        return "<Paragraph with {0} sentences, {1} words>".format(
            len(self.sentences),
            len(self.words),
        )

    def __repr__(self):
        return self.__str__()
