# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import math

from pprint import pformat
from collections import Sequence, Counter
from copy import deepcopy
from ..utils import to_unicode

string_types = (str, unicode,)


class TfDocumentModel(object):
    """Term-Frequency document model (term = word)."""
    def __init__(self, words, tokenizer=None):
        if isinstance(words, string_types) and tokenizer is None:
            raise ValueError(
                "Tokenizer has to be given if ``words`` is not a sequence.")
        elif isinstance(words, string_types):
            words = tokenizer.to_words(to_unicode(words))
        elif not isinstance(words, Sequence):
            raise ValueError(
                "Parameter ``words`` has to be sequence or string with tokenizer given.")

        self._terms = Counter(map(unicode.lower, words))
        self._max_frequency = max(self._terms.values()) if self._terms else 1
        self._all_frequency = sum(self._terms.values()) if self._terms else 1
        # normalize the tf value of terms
        if self._terms:
            self._normalized_tf_terms = deepcopy(self._terms)
            self._normalized_with_maxTf = deepcopy(self._terms)
            for word, tf_value in self._terms.items():
                self._normalized_tf_terms[word] = tf_value / self._all_frequency
                self._normalized_with_maxTf[word] = tf_value / self._max_frequency

    @property
    def magnitude(self):
        """
        Lenght/norm/magnitude of vector representation of document.
        This is usually denoted by ||d||.
        """
        return math.sqrt(sum(t**2 for t in self._terms.values()))

    @property
    def terms(self):
        return self._terms.keys()

    def most_frequent_terms(self, count=0):
        """
        Returns ``count`` of terms sorted by their frequency
        in descending order.

        :parameter int count:
            Max. number of returned terms. Value 0 means no limit (default).
        """
        # sort terms by number of occurrences in descending order
        terms = sorted(self._terms.items(), key=lambda i: -i[1])

        terms = tuple(i[0] for i in terms)
        if count == 0:
            return terms
        elif count > 0:
            return terms[:count]
        else:
            raise ValueError(
                "Only non-negative values are allowed for count of terms.")

    def term_frequency(self, term):
        """
        Returns frequency of term in document.

        :returns int:
            Returns count of words in document.
        """
        return self._terms.get(term, 0)

    def term_frequency_normalized(self, term, method="sum"):
        """
        Returns normalized frequency of term in document.

        :parameter method:
            normalized with max or sum of the tf_value
        :returns float:
            Returns normalized count of words in document.
        """
        if method.startswith("s"):
            return self._normalized_tf_terms.get(term, 0.0)
        elif method.startswith("m"):
            return self._normalized_with_maxTf.get(term, 0.0)
        else:
            raise ValueError(
                "method should be sum or max"
            )

    def normalized_term_frequency(self, term, smooth=0.0):
        """
        Returns normalized frequency of term in document.
        http://nlp.stanford.edu/IR-book/html/htmledition/maximum-tf-normalization-1.html

        :parameter float smooth:
            0.0 <= smooth <= 1.0, generally set to 0.4, although some
            early work used the value 0.5. The term is a smoothing term
            whose role is to damp the contribution of the second term.
            It may be viewed as a scaling down of TF by the largest TF
            value in document.
        :returns float:
            0.0 <= frequency <= 1.0, where 0 means no occurence in document
            and 1 the most frequent term in document.
        """
        frequency = self.term_frequency(term) / self._max_frequency
        return smooth + (1.0 - smooth)*frequency

    def __repr__(self):
        return "<TfDocumentModel %s>" % pformat(self._terms)
