# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from ._summarizer import AbstractSummarizer
from nlp_sum.my_sum.models import DocumentSet, Document

import random
import math
import re

import pulp
import numpy

from collections import defaultdict, deque, Counter
from numpy.linalg import norm
from nltk.stem import SnowballStemmer



class State:
    """ State class

    Internal class used as a structure to keep track of the search state in
    the tabu_search method.

    Args:
        subset (set): a subset of sentences
        concepts (Counter): a set of concepts for the subset
        length (int): the length in words
        score (int): the score for the subset
    """
    def __init__(self):
        self.subset = set()
        self.concepts = Counter()
        self.length = 0
        self.score = 0


class conceptILPSummarizer(AbstractSummarizer):

    _stop_words = frozenset()
    WORD_PATTERN = re.compile(r"^[^\W\d_]+$", re.UNICODE)

    def __init__(self, language="english", stemmer_or_not=False):
        if language.startswith("en") and stemmer_or_not:
            super(conceptILPSummarizer, self).__init__(
                language,
                SnowballStemmer("english").stem
            )
        else:
            super(conceptILPSummarizer, self).__init__(language)

        if self.language == "english":
            self._join_ngram = lambda ngram : ' '.join(ngram)
        if self.language == "chinese":
            self._join_ngram = lambda ngram : ''.join(ngram)

        self.weights = {}
        self.sentences = []
        self.concept2sent = defaultdict(set)
        self.concept_set = defaultdict(frozenset)
        self.word2sent = defaultdict(set)
        self.word_frequency = defaultdict(int)

    @property
    def stop_words(self):
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self,
                 document_set,
                 words_limit=100,
                 ngram=2,
                 prune_sen=True,
                 prune_method="top-n",
                 para=100,
                 method="ilp",
                 summary_order="origin"):

        self.sentences = document_set.sentences
        self.summary_order = summary_order
        self._extract_ngrams(ngram)
        self._compute_document_frequency_for_concept(document_set)
        self._compute_word_frequency(document_set)
        # if you want to use threshold, be careful because you
        # the frequency of most of concept is low
        if prune_sen:
            self._prune_sentences()
        if prune_method:
            self._prune_concepts(prune_method.lower(), para)
        self._compute_concept2sent()
        self._compute_concept_sets()

        if method.lower() == "greedy":
            value, summary_idx_set = self._greedy_approximation(words_limit)
        elif method.lower() == "tabu":
            value, summary_idx_set = self._tabu_search(words_limit)
        elif method.lower() == "ilp":
            value, summary_idx_set = self._solve_ilp_problem(words_limit)

        # sorted summary_idx_set in ascending order
        if self.summary_order == "origin":
            summary_idx_set = sorted(summary_idx_set)
        summary = tuple(self.sentences[idx] for idx in summary_idx_set)
        return summary

    @staticmethod
    def is_word(word):
        return bool(conceptILPSummarizer.WORD_PATTERN.search(word))

    def _filter_out_stop_words(self, words):
        return [word for word in words if word not in self._stop_words]

    def _normalize_words(self, words):
        words = map(self.normalize_word, words)
        return self._filter_out_stop_words(words)

    def _extract_ngrams(self, n=2):
        """
        Extract the ngrams of words from the input sentences.
        :para n(int): the number of words for ngrams
        """
        for i, sentence in enumerate(self.sentences):
            text_length = len(sentence.words)
            max_index_ngram = text_length - n
            sentence.concepts = []
            for j in xrange(max_index_ngram + 1):
                ngram = []

                for k in xrange(j, j+n):
                    ngram.append(sentence.words[k])

                # do not consider ngrams containing non-word
                is_word_set = map(conceptILPSummarizer.is_word, ngram)
                if not all(is_word_set):
                    continue

                # do not consider ngrams composed of only stop_words
                stops = [term for term in ngram if term in self._stop_words]
                if len(stops) == len(ngram):
                    continue

                ngram = map(self.normalize_word, ngram)
                #add the ngram to the concepts
                sentence.concepts.append(self._join_ngram(ngram))

    def _compute_document_frequency_for_concept(self, document_set):
        if isinstance(document_set, Document):
            document_set = DocumentSet([document_set])
        concept_in_every_doc = [set() for doc in document_set.documents]
        for idx, doc in enumerate(document_set.documents):
            for sentence in doc.sentences:
                for concept in sentence.concepts:
                    concept_in_every_doc[idx].add(concept)

        concept_in_every_doc = [list(concept_set) for concept_set in concept_in_every_doc]

        for doc_concepts in concept_in_every_doc:
            for concept in doc_concepts:
                if concept not in self.weights:
                    number_doc = sum(1 for doc in concept_in_every_doc
                                     if concept in doc)
                    self.weights[concept] = number_doc

    def _compute_word_frequency(self, document_set):
        """compute the frequency of each word in document_set"""

        for i, sentence in enumerate(document_set.sentences):
            for word in self._normalize_words(sentence.words):
                self.word2sent[word].add(i)
                self.word_frequency[word] += 1

    def _prune_sentences(self,
                         mininum_sentence_length=10,
                         remove_citations=True,
                         remove_redundancy=True):
        """Prune the sentences.

        Remove the sentences that are shorter than a given length, redundant
        sentences and citations from entering the summary.

        Args:
            mininum_sentence_length (int): the minimum number of words for a
              sentence to enter the summary, defaults to 5
            remove_citations (bool): indicates that citations are pruned,
              defaults to True
            remove_redundancy (bool): indicates that redundant sentences are
              pruned, defaults to True

        """
        pruned_sentences = []

        # loop over the sentences
        for sentence in self.sentences:

            # prune short sentences
            if self._get_sentence_length(sentence) < mininum_sentence_length:
                continue

            # sometimes the words of sentence may be empty set
            if not sentence.words:
                continue

            # prune citations
            first_token, last_token = sentence.words[0], sentence.words[-1]
            if remove_citations and \
               (first_token == u"``" or first_token == u'"') and \
               (last_token == u"''" or first_token == u'"'):
                continue

            # prune identical and almost identical sentences
            if remove_redundancy:
                is_redundant = False
                for prev_sentence in pruned_sentences:
                    if sentence.words == prev_sentence.words:
                        is_redundant = True
                        break

                if is_redundant:
                    continue
            # otherwise add the sentence to the pruned sentence container
            pruned_sentences.append(sentence)
        self.sentences = pruned_sentences

    def _prune_concepts(self, method="thresold", para=3):
        """Prune the concepts for efficient summarization.
        Args:
            method (str): the method for pruning concepts that can be whether
              by using a minimal value for concept scores (threshold) or using
              the top-N highest scoring concepts (top-n), defaults to
              threshold.
            value (int): the value used for pruning concepts, defaults to 3.
        """
        # 'threshold' pruning method
        if method == "threshold":
            concepts = self.weights.keys()
            for concept in concepts:
                if self.weights[concept] < para:
                    del self.weights[concept]

        # 'top-n' pruning method
        elif method == "top-n":
            sorted_concepts = sorted(self.weights,
                                     key=lambda x: self.weights[x],
                                     reverse=True)
            concepts = self.weights.keys()
            for concept in concepts:
                if concept not in sorted_concepts[:para]:
                    del self.weights[concept]

        for i, sentence in enumerate(self.sentences):
            concepts = sentence.concepts
            sentence.concepts = [c for c in concepts
                                 if c in self.weights]

    def _compute_concept2sent(self):
        """Compute the inverted concept to sentences dictionary. """
        for i, sentence in enumerate(self.sentences):
            for concept in sentence.concepts:
                self.concept2sent[concept].add(i)

    def _compute_concept_sets(self):
        """compute the concept sets for each sentence."""
        for i, sentence in enumerate(self.sentences):
            for concept in sentence.concepts:
                self.concept_set[i] |= {concept}

    def _greedy_approximation(self, words_limit=100):
        """Greedy approximation of the ILP model.

        Args:
            words_limit (int): the maximum size in words of the summary,
              defaults to 100.

        Returns:
            (value, set) tuple (int, list): the value of the approximated
              objective function and the set of selected sentences as a tuple.

        """
        # initialize the inverted concept2sent dictionary if not already created
        if not self.concept2sent:
            self._compute_concept2sent()

        # initialize weights
        weights ={}

        # initialize the score of the best singleton
        best_singleton_score = 0
        best_singleton_idx = -1
        sentences_count = len(self.sentences)
        # compute initial weights and fill the reverse index
        # while keeping the track of the best singleton solutin
        for idx, sentence in enumerate(self.sentences):
            weights[idx] = sum(self.weights[c] for c in set(sentence.concepts))
            sent_len = self._get_sentence_length(sentence)
            if sent_len <= words_limit and weights[idx] > best_singleton_score:
                best_singleton_score = weights[idx]
                best_singleton_idx = idx

        # initialize the selected solution properties
        select_subset, select_concepts = set(), set()
        select_length, select_score = 0, 0
        # greedily select a sentence
        while True:
            ###################################################################
            # RETRIEVE THE BEST SENTENCE
            ###################################################################

            # sort the sentences by gain and punish length
            sent_weight = ((weights[idx] / self._get_sentence_length(sentence),
                            -self._get_sentence_length(sentence),
                            idx)
                           for idx, sentence in enumerate(self.sentences))
            sort_sent = sorted(sent_weight, reverse=True)

            # select the first sentence that satisfy length limit
            find_sent = False
            for weight, reverse_length, sent_idx in sort_sent:
                summary_len = select_length - reverse_length
                if summary_len <= words_limit:
                    find_sent = True
                    break
            # if we don't find a sentence, break out of the main while loop
            if not find_sent:
                break
            # if the gain is null, break out of the main while loop
            if not weights[sent_idx]:
                break
            # update the selected subset
            select_subset.add(sent_idx)
            select_score += weights[sent_idx]
            select_length += (-reverse_length)

            # update sentence weights with the reverse index
            for concept in set(self.sentences[sent_idx].concepts):
                if concept not in select_concepts:
                    for sentence_idx in self.concept2sent[concept]:
                        weights[sentence_idx] -= self.weights[concept]

            # update the last selected subset property
            select_concepts.update(self.sentences[sent_idx].concepts)

        # check if a single sentence has a better score than the greedy solution
        if best_singleton_score > select_score:
            return best_singleton_score, set([best_singleton_idx])
        # return the (objective function value, solution) tuple
        return select_score, select_subset

    def _select_sentences(self,
                          words_limit,
                          weights,
                          state,
                          tabu_set,
                          mutation_group):
        """Greedy sentence selector.
        Args:
            words_limit (int): the maximum size in words of the summary,
              defaults to 100.
            weights (dictionary): the sentence weights dictionary. This
              dictionnary is updated during this method call (in-place).
            state (State): the state of the tabu search from which to start
              selecting sentences.
            tabu_set (iterable): set of sentences that are tabu: this
              selector will not consider them.
            mutation_group (boolean): flag to consider the mutations as a
              group: we'll check sentence combinations in the tabu list, not
              sentences alone.

        Returns:
            state (State): the new state of the search. Also note that
              weights is modified in-place.

        """
        # greedily select a sentence while respecting the tabu
        while True:
            ###################################################################
            # RETRIEVE THE BEST SENTENCE
            ###################################################################
            sent_weight = ((weights[idx] / self._get_sentence_length(sentence),
                            -self._get_sentence_length(sentence),
                            idx)
                           for idx, sentence in enumerate(self.sentences)
                           if self._get_sentence_length(sentence) + state.length <= words_limit)
            sort_sent = sorted(sent_weight, reverse=True)

            # select the first sentence that satisfy the length limit
            for weight, reverse_length, sent_idx in sort_sent:
                if mutation_group:
                    subset = state.subset | {sent_idx}
                    for tabu in tabu_set:
                        # test if tabu is in subset
                        if tabu <= subset:
                            break
                    else:
                        break
                elif sent_idx not in tabu_set:
                        break
            # if we don't find a sentence, break out of the main while loop
            else:
                break
            # if the gain is null, break out of the main while loop
            if not weights[sent_idx]:
                break
            # update state
            state.subset = state.subset | {sent_idx}
            state.concepts.update(self.concept_set[sent_idx])
            state.length += (-reverse_length)
            state.score += weights[sent_idx]
            # update sentence weights with the reverse index
            for concept in set(self.concept_set[sent_idx]):
                if state.concepts[concept] == 1:
                    for sentence_idx in self.concept2sent[concept]:
                        weights[sentence_idx] -= self.weights[concept]

        return state

    def _unselect_sentences(self, weights, state, to_remove):
        """Sentence ``un-selector'' (reverse operation of the
          select_sentences method).

        Args:
            weights (dictionary): the sentence weights dictionary. This
              dictionnary is updated during this method call (in-place).
            state (State): the state of the tabu search from which to start
              un-selecting sentences.
            to_remove (iterable): set of sentences to unselect.

        Returns:
            state (State): the new state of the search. Also note that
              weights is modified in-place.

        """
        # remove the sentence indices from the solution subset
        state.subset -= to_remove
        for sent_idx in to_remove:
            # update state
            state.concepts.subtract(self.concept_set[sent_idx])
            state.length -= self._get_sentence_length(self.sentences[sent_idx])
            # update sentence weights with reverse index
            for concept in set(self.concept_set[sent_idx]):
                if not state.concepts[concept]:
                    for sentence_index in self.concept2sent[concept]:
                        weights[sentence_index] += self.weights[concept]
            state.score -= weights[sent_idx]

        return state

    def _tabu_search(self,
                     words_limit=100,
                     memory_size=10,
                     iterations=100,
                     mutation_size=2,
                     mutation_group=True):
        """Greedy approximation of the ILP model with a tabu search
          meta-heuristic.

        Args:
            words_limit (int): the maximum size in words of the summary,
              defaults to 100.
            memory_size (int): the maximum size of the pool of sentences
              to ban at a given time, defaults at 5.
            iterations (int): the number of iterations to run, defaults at
              30.
            mutation_size (int): number of sentences to unselect and add to
              the tabu list at each iteration.
            mutation_group (boolean): flag to consider the mutations as a
              group: we'll check sentence combinations in the tabu list, not
              sentences alone.
        Returns:
            (value, set) tuple (int, list): the value of the approximated
              objective function and the set of selected sentences as a tuple.

        """
        if not self.concept2sent:
            self._compute_concept2sent()
        if not self.concept_set:
            self._compute_concept_sets()

        # initialize the weights
        weights = {}
        # initialize the score of the best singleton
        best_singleton_score = 0
        best_singleton_idx = -1
        sentences_count = len(self.sentences)
        # compute initial weights and fill the reverse index
        # while keeping track of the best singleton solution
        for idx, sentence in enumerate(self.sentences):
            weights[idx] = sum(self.weights[c] for c in set(sentence.concepts))
            sent_len = self._get_sentence_length(sentence)
            if sent_len <= words_limit and weights[idx] > best_singleton_score:
                best_singleton_score = weights[idx]
                best_singleton_idx = idx

        best_subset, best_score = None, 0
        state = State()

        for i in xrange(iterations):
            queue = deque([], memory_size)
            # greedily select sentences
            state = self._select_sentences(words_limit,
                                           weights,
                                           state,
                                           queue,
                                           mutation_group)

            if state.score > best_score:
                best_subset = state.subset.copy()
                best_score = state.score
            to_tabu = set(random.sample(state.subset, mutation_size))
            state = self._unselect_sentences(weights, state, to_tabu)
            queue.extend(to_tabu)

        # check if a singleton has a better score than our greedy solution
        if best_singleton_score > best_score:
            return best_singleton_score, set([best_singleton_idx])

        # returns the (objective function value, solution) tuple
        return best_score, best_subset

    def _solve_ilp_problem(self,
                           words_limit=100,
                           solver='glpk',
                           excluded_solutions=[],
                           unique=False):
        """Solve the ILP formulation of the concept-based model.
        Args:
            words_limit (int): the maximum size in words of the summary,
              defaults to 100.
            solver (str): the solver used, defaults to glpk.
            excluded_solutions (list of list): a list of subsets of sentences
              that are to be excluded, defaults to []
            unique (bool): modify the model so that it produces only one optimal
              solution, defaults to False
        Returns:
            (value, set) tuple (int, list): the value of the objective function
              and the set of selected sentences as a tuple.
        """
        # initialize container shortcuts
        concepts = self.weights.keys()
        W = self.weights
        L = words_limit
        C = len(concepts)
        S = len(self.sentences)

        assert self.word_frequency, "word_frequency must be calculated first"

        words = self.word_frequency.keys()
        frequency = self.word_frequency
        words_count = len(words)

        concepts = sorted(self.weights, key=self.weights.get, reverse=True)

        # fomulation of the ILP problem
        prob = pulp.LpProblem("ILP for Summarization", pulp.LpMaximize)

        # initialize the concepts binary variables
        c = pulp.LpVariable.dicts(name='c',
                                  indexs=range(C),
                                  lowBound=0,
                                  upBound=1,
                                  cat='Integer')
        # initialize the sentences binary variables
        s = pulp.LpVariable.dicts(name='s',
                                  indexs=range(S),
                                  lowBound=0,
                                  upBound=1,
                                  cat='Integer')
        # initialize the word(term) binary variables
        t = pulp.LpVariable.dicts(name='t',
                                  indexs=range(words_count),
                                  lowBound=0,
                                  upBound=1,
                                  cat='Integer')
        # objective function
        if unique:
            prob += sum(W[concepts[i]] * c[i] for i in range(C)) + \
                    10e-6 * sum(frequency[words[j]] * t[j] for j in range(words_count))
        else:
            prob += sum(W[concepts[i]] * c[i] for i in range(C))

        # constraint for summary size
        prob += sum(s[i] * self._get_sentence_length(self.sentences[i])
                    for i in range(S)) <= L

        # integrity constraint
        for i in range(C):
            for j in range(S):
                if concepts[i] in self.sentences[j].concepts:
                    prob += s[j] <= c[i]

        for i in range(C):
            prob += sum(s[j] for j in range(S)
                        if concepts[i] in self.sentences[j].concepts) >= c[i]

        # word integrity constraint
        if unique:
            for i in range(words_count):
                for j in self.word2sent[words[i]]:
                    prob += s[j] <= t[i]

            for i in range(words_count):
                prob += sum(s[j] for j in self.word2sent[words[i]]) >= t[i]

        # constraint for finding optimal solutions
        for sentence_set in excluded_solutions:
            prob += sum(s[i] for i in sentence_set) <= len(sentence_set) - 1

        # solve the ilp problem
        if solver.lower() == "glpk":
            prob.solve(pulp.GLPK(msg=0))
        elif solver.lower() == "gurobi":
            prob.solve(pulp.GUROBI(msg=0))
        elif solver.lower() == "cplex":
            prob.solve(pulp.CPLEX(msg=0))
        else:
            raise AssertionError("no solver specified")

        # retrieve the optimal subset of sentences
        solution = set([idx for idx in range(S) if s[idx].varValue == 1])

        # return the (objective function value, solution) tuple
        return (pulp.value(prob.objective), solution)
