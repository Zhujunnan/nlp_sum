# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from itertools import chain
from ..utils import cached_property, to_unicode, unicode_compatible


@unicode_compatible
class DocumentSet(object):
    # __slots__ = (
    #     "_documents",
    #     "_cached_property_documents",
    #     "_cached_property_paragraphs"
    #     "_cached_property_sentences",
    #     "_cached_property_words",
    # )

    def __init__(self, documents):
        self._documents = tuple(documents)

    @cached_property
    def documents(self):
        return self._documents

    @cached_property
    def paragraphs(self):
        paragraphs = (d.paragraphs for d in self._documents)
        return tuple(chain(*paragraphs))

    @cached_property
    def sentences(self):
        sentences = (d.sentences for d in self._documents)
        return tuple(chain(*sentences))

    @cached_property
    def words(self):
        words = (d.words for d in self._documents)
        return tuple(chain(*words))

    def __unicode__(self):
        return ("<Docset with {0} docments, {1} paragraphs, {2} sentences" +
                ", {3} words>").format(
                    len(self.documents),
                    len(self.paragraphs),
                    len(self.sentences),
                    len(self.words),
                )

    def __str__(self):
        return self.__str__()
