"""
Module for functions that work strings, i.e. single tokens.

.. codeauthor:: Markus Konrad <markus.konrad@wzb.eu>
"""
from __future__ import annotations

import math
import unicodedata
from html.parser import HTMLParser
from typing import Optional

from bidict import bidict


OOV = 0
DOC_START = 5
DOC_END = 6
SENT_START = 10
SENT_END = 11
SPECIAL_TOKENS = bidict({
    DOC_START: '<d>',
    DOC_END: '</d>',
    SENT_START: '<s>',
    SENT_END: '</s>',
    OOV: '<oov>'
})


#%% functions that operate on single string tokens or texts

def numbertoken_to_magnitude(numbertoken: str, char: str = '0', firstchar: str = '1', below_one: str = '0',
                             zero: str = '0', decimal_sep: str = '.', thousands_sep: str = ',',
                             drop_sign: bool = False, value_on_conversion_error: Optional[str] = '') -> str:
    """
    Convert a string token `numbertoken` that represents a number (e.g. "13", "1.3" or "-1313") to a string token that
    represents the magnitude of that number by repeating `char` ("10", "1", "-1000" for the mentioned examples). A
    different first character can be set via `firstchar`. The sign can be dropped via `drop_sign`.

    If `numbertoken` cannot be converted to a float, either the value `value_on_conversion_error` is returned or
    `numbertoken` is returned unchanged if `value_on_conversion_error` is None.

    :param numbertoken: token that represents a number
    :param char: character string used to represent single orders of magnitude
    :param firstchar: special character used for first character in the output
    :param below_one: special character used for numbers with absolute value below 1 (would otherwise return `''`)
    :param zero: if `numbertoken` evaluates to zero, return this string
    :param decimal_sep: decimal separator used in `numbertoken`; this is language-specific
    :param thousands_sep: thousands separator used in `numbertoken`; this is language-specific
    :param drop_sign: if True, drop the sign in number `numbertoken`, i.e. use absolute value
    :param value_on_conversion_error: determines return value when `numbertoken` cannot be converted to a number;
                                      if None, return input `numbertoken` unchanged, otherwise return
                                      `value_on_conversion_error`
    :return: string that represents the magnitude of the input or an empty string
    """
    if decimal_sep != '.':
        numbertoken = numbertoken.replace(decimal_sep, '.')

    if thousands_sep:
        numbertoken = numbertoken.replace(thousands_sep, '')

    try:
        number = float(numbertoken)
    except ValueError:  # catches float conversion error
        if value_on_conversion_error is None:
            return numbertoken
        else:
            return value_on_conversion_error

    prefix = '-' if not drop_sign and number < 0 else ''
    abs_number = abs(number)

    if abs_number < 1:
        return prefix + below_one

    try:
        magn = math.floor(math.log10(abs_number)) + 1    # absolute magnitude, sign is discarded here
    except ValueError:  # catches domain error when taking log10(0)
        return zero

    if firstchar != char:
        return prefix + firstchar + char * (magn-1)
    else:
        return prefix + char * magn


def simplify_unicode_chars(token: str, method: str = 'icu', ascii_encoding_errors: str = 'ignore') -> str:
    """
    *Simplify* unicode characters in string `token`, i.e. remove diacritics, underlines and
    other marks. Requires `PyICU <https://pypi.org/project/PyICU/>`_ to be installed when using
    ``method="icu"``.

    :param docs: a Corpus object
    :param token: string to simplify
    :param method: either ``"icu"`` which uses `PyICU <https://pypi.org/project/PyICU/>`_ for "proper"
                   simplification or ``"ascii"`` which tries to encode the characters as ASCII; the latter
                   is not recommended and will simply dismiss any characters that cannot be converted
                   to ASCII after decomposition
    :param ascii_encoding_errors: only used if `method` is ``"ascii"``; what to do when a character cannot be
                                  encoded as ASCII character; can be either ``"ignore"`` (default – replace by empty
                                  character), ``"replace"`` (replace by ``"???"``) or ``"strict"`` (raise a
                                  ``UnicodeEncodeError``)
    :return: simplified string
    """

    method = method.lower()
    if method == 'icu':
        try:
            from icu import UnicodeString, Transliterator, UTransDirection
        except ImportError:
            raise RuntimeError('package PyICU (https://pypi.org/project/PyICU/) must be installed to use this method')

        u = UnicodeString(token)
        trans = Transliterator.createInstance("NFD; [:M:] Remove; NFC", UTransDirection.FORWARD)
        trans.transliterate(u)
        return str(u)
    elif method == 'ascii':
        return unicodedata.normalize('NFKD', token).encode('ASCII', errors=ascii_encoding_errors).decode('utf-8')
    else:
        raise ValueError('`method` must be either "icu" or "ascii"')


def strip_tags(value: str) -> str:
    """
    Return the given HTML with all tags stripped and HTML entities and character references converted to Unicode
    characters.

    Code taken and adapted from https://github.com/django/django/blob/main/django/utils/html.py.

    :param value: input string
    :return: string without HTML tags
    """
    # Note: in typical case this loop executes _strip_once once. Loop condition
    # is redundant, but helps to reduce number of executions of _strip_once.
    value = str(value)
    while '<' in value and '>' in value:
        new_value = _strip_once(value)
        if value.count('<') == new_value.count('<'):
            # _strip_once wasn't able to detect more tags.
            break
        value = new_value
    return value



#%% helper functions and classes


class _MLStripper(HTMLParser):
    """
    Code taken and adapted from https://github.com/django/django/blob/main/django/utils/html.py.
    """
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.reset()
        self.fed = []

    def handle_data(self, d):
        self.fed.append(d)

    def get_data(self):
        return ''.join(self.fed)


def _strip_once(value):
    """
    Internal tag stripping utility used by strip_tags.

    Code taken and adapted from https://github.com/django/django/blob/main/django/utils/html.py.
    """
    s = _MLStripper()
    s.feed(value)
    s.close()
    return s.get_data()
