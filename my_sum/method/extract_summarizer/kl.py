# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import math

from nltk.stem import SnowballStemmer
from ._summarizer import AbstractSummarizer


class KLSummarizer(AbstractSummarizer):
    """
    Method that greedily adds sentences to a summary so long as it decreases the
    KL Divergence.
    Source: http://www.aclweb.org/anthology/N09-1041
    """

    _stop_words = frozenset()

    def __init__(self, language="english", stemmer_or_not=False):
        # super(KLSummarizer, self).__init__(language)
        if language.startswith("en") and stemmer_or_not:
            super(KLSummarizer, self).__init__(
                language,
                SnowballStemmer("english").stem
            )
        else:
            super(KLSummarizer, self).__init__(language)

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, documentSet, words_limit):
        ratings = self._get_ratings(documentSet)
        return self._get_best_sentences(documentSet.sentences, words_limit, ratings)

    def _get_ratings(self, document):
        """
        'document' can be document or document_set
        """
        sentences = document.sentences
        ratings = self._compute_ratings(sentences)
        return ratings

    def _get_all_words_in_doc(self, sentences):
        return [word for sent in sentences for word in sent.words]

    def _get_content_words_in_sentence(self, sentence):
        normalized_words = self._normalize_words(sentence.words)
        normalized_content_words = self._filter_out_stop_words(normalized_words)
        return normalized_content_words

    def _get_all_content_words_in_doc(self, sentences):
        all_words = self._get_all_words_in_doc(sentences)
        content_words = self._filter_out_stop_words(all_words)
        normalized_content_words = self._normalize_words(content_words)
        return normalized_content_words

    def _normalize_words(self, words):
        return [self.normalize_word(word) for word in words]

    def _filter_out_stop_words(self, words):
        return [word for word in words if word not in self._stop_words]

    def _compute_word_freq(self, word_list):
        word_freq = {}
        for word in word_list:
            word_freq[word] = word_freq.get(word, 0) + 1
        return word_freq

    def compute_tf(self, sentences):
        """
        Computes the normalized term frequency as explained in http://www.tfidf.com/

        :type sentences: [nlp_sum.my_sum.models.Sentence]
        """
        content_words = self._get_all_content_words_in_doc(sentences)
        content_words_count = len(content_words)
        content_words_freq = self._compute_word_freq(content_words)
        content_words_tf = dict(
            (word, freq / content_words_count)
            for word, freq in content_words_freq.items()
        )
        return content_words_tf

    def _joint_freq(self, word_list1, word_list2):
        # :type word_list1 word_list2: dict
        # calculate total number of words
        total_len = len(word_list1) + len(word_list2)

        # word frequencies within each list
        word_freq1 = self._compute_word_freq(word_list1)
        word_freq2 = self._compute_word_freq(word_list2)

        joint = word_freq1.copy()

        for word in word_freq2:
            if word in joint:
                joint[word] += word_freq2[word]
            else:
                joint[word] = word_freq2[word]

        # normalize the tf by the total number
        for word in joint:
            joint[word] /= total_len
        return joint

    def _kl_divergence(self, summary_freq, doc_freq):
        """
        Note: Could import scipy.stats and use scipy.stats.entropy(doc_freq, summary_freq)
        but this gives equivalent value without the import
        """
        value = 0
        for word in summary_freq:
            frequency = doc_freq.get(word, 0)
            if frequency:
                value += frequency * math.log(frequency / summary_freq[word])
        return value

    def _compute_ratings(self, sentences):
        word_freq = self.compute_tf(sentences)
        ratings = {}
        summary = []

        # make it a list so that it can be modified
        sentences_list = list(sentences)

        # get all content words once for efficiency
        sentences_as_words = [
            self._get_content_words_in_sentence(sent)
            for sent in sentences
        ]

        # remove one sentence per iteration by adding to summary
        while len(sentences_list) > 0:
            # store all the kl value
            kl_value = []

            # converts summary to word list
            summary_as_word_list = self._get_all_content_words_in_doc(summary)

            for sent_word in sentences_as_words:
                # calculate the joint frequency through combining the word lists
                joint_freq = self._joint_freq(sent_word, summary_as_word_list)

                # adds the calculated kl divergence to the list in index = sentence used
                kl_value.append(self._kl_divergence(joint_freq, word_freq))

            # to consider and then add it to summary
            indexToRemove = self._find_index_of_best_sentence(kl_value)
            best_sentence = sentences_list.pop(indexToRemove)
            del sentences_as_words[indexToRemove]
            summary.append(best_sentence)

            # value is the iteration in which it was removed multiplied by -1
            # so that the first removed (the most important) have highest value
            ratings[best_sentence] = -1 * len(ratings)

        return ratings

    def _find_index_of_best_sentence(self, kl_value):
        """
        best_sentence has the smallest kl_divergence
        """
        return kl_value.index(max(kl_value))
