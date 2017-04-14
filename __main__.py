# -*- coding: utf-8 -*-

"""
nlp_sum - multi-document summarizer.

Usage:
    nlp_sum (ilp | kl | lexrank | lsa | random | submodular | textrank) [--length=<length>] [--language=<lang>] [--stopwords=<file_path>] [--stem] [--format=<format>] [--para=<parameter>] [--output=<file_path>] --file=<file_path>
    nlp_sum --help

Options:
    --length=<length>        Length limit of summarized text.
    --language=<lang>        Natural language of summarized text. [default: english]
    --stopwords=<file_path>  Path to a file containing a list of stopwords. One word per line in UTF-8 encoding.
                             If it is not specified default list of stopwords is used according to chosen lang.
    --stem                   If specified, will need stem.
    --format=<format>        Format of input document. Possible values: plaintext
    --file=<file_path>       Path to the text file to summarize.(directory of file path)
    --output=<file_path>     File path to write the result of summarization.
    --para=<parameter>       parameter to the summarizer in string format such as '0.1 0.2'
    --help, -h               Displays current application version.

"""

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals


from os.path import isfile, isdir, abspath
from docopt import docopt
from nlp_sum.my_sum.utils import to_string, get_stop_words, read_stop_words
from nlp_sum.my_sum.nlp.Tokenizer import Tokenizer
from nlp_sum.my_sum.parse.plaintext import PlaintextParser
from nlp_sum.my_sum.method.extract_summarizer.conceptILP import conceptILPSummarizer
from nlp_sum.my_sum.method.extract_summarizer.kl import KLSummarizer
from nlp_sum.my_sum.method.extract_summarizer.lexrank import LexRankSummarizer
from nlp_sum.my_sum.method.extract_summarizer.lsa import LsaSummarizer
from nlp_sum.my_sum.method.extract_summarizer.random import RandomSummarizer
from nlp_sum.my_sum.method.extract_summarizer.submodular import SubmodularSummarizer
from nlp_sum.my_sum.method.extract_summarizer.textrank import TextRankSummarizer


PARSERS = {
    "plaintext" : PlaintextParser
}

METHODS = {
    "ilp" : conceptILPSummarizer,
    "kl" : KLSummarizer,
    "lexrank" : LexRankSummarizer,
    "lsa" : LsaSummarizer,
    "random" : RandomSummarizer,
    "submodular" : SubmodularSummarizer,
    "textrank" : TextRankSummarizer,
}

def handle_arguments(args):
    document_format = args['--format']
    if document_format is not None and document_format not in PARSERS:
        raise ValueError("Unsupported input format. Possible values are {0}. Given: {1}.").format(
            ", ".join(PARSERS.keys()),
            document_format
        )
    parser = PARSERS[document_format or "plaintext"]

    words_limit = args['--length'] or 250
    words_limit = int(words_limit)
    language = args['--language'] or "english"
    parser = parser(language)
    tokenizer = Tokenizer(language)

    if args['--file'] is not None:
        file_path = args['--file']
        file_path = abspath(file_path)
        if isdir(file_path):
            document_set = parser.build_documentSet_from_dir(
                file_path
            )
        elif isfile(file_path):
            document_set = parser.build_document_from_file(
                tokenizer, file_path
            )
        else:
            raise ValueError("Input file is invalid")

    if args['--stopwords']:
        stop_words = read_stop_words(args['--stopwords'])
    else:
        stop_words = get_stop_words(language)

    if args['--stem']:
        stem_or_not = True
    else:
        stem_or_not = False

    summarizer_class = next(cls for name, cls in METHODS.items() if args[name])
    summarizer = build_summarizer(summarizer_class, language, stop_words, stem_or_not)

    return document_set, summarizer, language, words_limit


def build_summarizer(summarizer_class, language, stop_words, stem_or_not):
    summarizer = summarizer_class(language, stem_or_not)
    summarizer.stop_words = stop_words
    return summarizer

def main(args=None):
    args = docopt(to_string(__doc__))
    document_set, summarizer, language, words_limit = handle_arguments(args)

    output_path = args['--output']

    summary = summarizer(document_set, words_limit)
    if language.startswith("en"):
        summary_text = ' '.join(sentence._texts for sentence in summary)
    elif language.startswith("ch"):
        summary_text = ''.join(sentence._texts + '。' for sentence in summary)

    if output_path:
        output_path = abspath(output_path)
        with open(output_path, 'wb') as file:
            file.write(summary_text.encode("utf8"))
    else:
        print(summary_text)

    return 0

if __name__ == "__main__":
    # args = docopt(to_string(__doc__))
    # print(args)
    try:
        exit_code = main()
        exit(exit_code)
    except KeyboardInterrupt:
        exit(1)
    except Exception as e:
        print(e)
        exit(1)
