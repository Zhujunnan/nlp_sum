# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from functools import wraps
from os.path import abspath, dirname

def to_unicode(unicode_or_str):
    """receive str or unicode and always return unicode """
    if isinstance(unicode_or_str, str):
        value = unicode_or_str.decode("utf-8")
    else:
        value = unicode_or_str
    return value

def to_string(unicode_or_str):
    """receive str or unicode and always return string"""
    if isinstance(unicode_or_str, unicode):
        value = unicode_or_str.encode("utf-8")
    else:
        value = unicode_or_str
    return value

def cached_property(getter):
    """
    Decorator that converts a method into memorized property.The decorator
    works as expected only for classes with attribute '__dict__' and immutable
    properties
    """
    @wraps(getter)
    def decorator(self):
        key = "_cached_property_" + getter.__name__

        if not hasattr(self, key):
            setattr(self, key, getter(self))

        return getattr(self, key)

    return property(decorator)

def unicode_compatible(cls):
    """
    Decorator for unicode compatible classes. Method '__unicode__' has to be
    implemented to work decorator as expected
    """
    cls.__str__ = lambda self: self.__unicode__().encode("utf-8")

    return cls

def get_stop_words(language):
    stop_words_path = abspath(dirname(__file__)) + "/data/stopwords/{0}.txt".format(language)
    # print(stop_words_path)
    stop_words_data = []
    try:
        with open(stop_words_path, 'rb') as file:
            for line in file:
                stop_words_data.append(line.splitlines()[0].strip())
    except IOError as e:
        raise LookupError("Stop-words are not availale for language {0}".format(language))
    return frozenset(to_unicode(word) for word in stop_words_data if word)




# if __name__ == "__main__":
#     stop_words = get_stop_words("chinese")
#     print(stop_words)
