# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

# from os.path import dirnam, join, abspath
from nlp_sum.my_sum.nlp.Tokenizer import Tokenizer
from nlp_sum.my_sum.utils import to_unicode
from nlp_sum.my_sum.models import DocumentSet, Document, Paragraph, Sentence


def build_document_cn(*sets_of_sentences):
    """sets_of_sentences can be a list or tuple of sentences in text format"""
    paragraphs = []

    for sentences in sets_of_sentences:
        paragraph_sentence = []
        for sentence_as_string in sentences:
            sentence = build_sentence(sentence_as_string, language="chinese")
            paragraph_sentence.append(sentence)
            paragraphs.append(Paragraph(paragraph_sentence))
    return Document(paragraphs)

def build_document_en(*sets_of_sentences):
    """sets_of_sentences can be a list or tuple of sentences in text format"""
    paragraphs = []

    for sentences in sets_of_sentences:
        paragraph_sentence = []
        for sentence_as_string in sentences:
            sentence = build_sentence(sentence_as_string, language="english")
            paragraph_sentence.append(sentence)
        paragraphs.append(Paragraph(paragraph_sentence))
    return Document(paragraphs)

def build_document_from_string(string="", language="english"):
    sentences = []
    paragraphs = []
    tokenizer = Tokenizer(language)

    for line in string.strip().splitlines():
        line = line.lstrip()
        if line:
            sentence_tuple = tokenizer.to_sentences(line)
            for sentence in sentence_tuple:
                sentences.append(build_sentence(sentence, language))
            paragraphs.append(Paragraph(sentences))
            sentences = []
        else:
            continue
    return Document(paragraphs)

def build_sentence(sentence_as_string="", language = "english"):
    tokenizer = Tokenizer(language)
    return Sentence(sentence_as_string, tokenizer)


# def build_document_from_string(string="", language="english"):
#     sentences = []
#     paragraphs = []

#     for line in string.strip().splitlines():
#         line = line.lstrip()
#         if line:
#             sentences.append(build_sentence(line, language))
#         else:
#             paragraphs.append(Paragraph(sentences))
#             sentences = []
#     paragraphs.append(Paragraph(sentences))
#     return Document(paragraphs)
