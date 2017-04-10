# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from itertools import chain
from ..utils import cached_property, unicode_compatible, to_unicode


@unicode_compatible
class Document(object):
    # __slots__ = (
    #     "_paragraphs",
    #     "cached_property_paragraphs"
    #     "_cached_property_sentences",
    #     "_cached_property_words",
    # )

    def __init__(self, paragraphs):
        """paragraphs should be a list or tuple"""
        self._paragraphs = tuple(paragraphs)

    @cached_property
    def paragraphs(self):
        return self._paragraphs

    @cached_property
    def sentences(self):
        sentences = (p._sentences for p in self._paragraphs)
        return tuple(chain(*sentences))

    @cached_property
    def words(self):
        words = (p.words for p in self._paragraphs)
        return tuple(chain(*words))

    def __unicode__(self):
        return ("<DOC with {0} paragraphs, {1} sentences, " +
                  "{2} words>").format(
                      len(self.paragraphs),
                      len(self.sentences),
                      len(self.words),
                  )

    def __repr__(self):
        return self.__str__()
