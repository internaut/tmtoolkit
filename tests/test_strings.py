"""
Tests for tmtoolkit.strings module.

.. codeauthor:: Markus Konrad <post@mkonrad.net>
"""
import string
from importlib.util import find_spec

import pytest
from hypothesis import given, strategies as st

from tmtoolkit.strings import numbertoken_to_magnitude, simplify_unicode_chars, strip_tags


@pytest.mark.parametrize('numbertoken, char, firstchar, below_one, drop_sign, expected', [
    ('', '0', '0', '0', True, ''),
    ('no number', '0', '0', '0', True, ''),
    ('0', '0', '0', '0', True, '0'),
    ('0.9', '0', '0', '0', True, '0'),
    ('0.1', '0', '0', '', True, ''),
    ('0.01', '0', '0', '0', True, '0'),
    ('-0.01', '0', '0', 'X', True, 'X'),
    ('1', '0', '0', '0', True, '0'),
    ('1', '0', '1', '0', True, '1'),
    ('10', '0', '0', '0', True, '00'),
    ('10', '0', '1', '0', True, '10'),
    ('123456', '0', '0', '0', True, '000000'),
    ('123456', '0', '1', '0', True, '100000'),
    ('123456', 'N', 'X', '0', True, 'XNNNNN'),
    ('123.456', '0', '0', '0', True, '000'),
    ('-123.456', '0', '0', '0', True, '000'),
    ('-123.456', '0', '0', '0', False, '-000'),
    ('-123.456', '0', '1', '0', False, '-100'),
    ('-0.0123', '0', '1', '0', False, '-0'),
    ('-1.0123', '0', '1', '0', False, '-1'),
    ('180,000', '0', '1', '0', False, '100000'),
    ('180,000.99', '0', '1', '0', False, '100000'),
])
def test_numbertoken_to_magnitude(numbertoken, char, firstchar, below_one, drop_sign, expected):
    res = numbertoken_to_magnitude(numbertoken, char=char, firstchar=firstchar, below_one=below_one,
                                   drop_sign=drop_sign)
    assert res == expected


@given(token=st.one_of(st.text(string.printable),
                       st.sampled_from(['\u00C7', '\u0043\u0327', '\u0043\u0332', 'é', 'ῷ'])),
       method=st.sampled_from(['icu', 'ascii', 'nonexistent']),
       ascii_encoding_errors=st.sampled_from(['ignore', 'replace']))
def test_simplify_unicode_chars(token, method, ascii_encoding_errors):
    if method == 'icu' and not find_spec('icu'):
        with pytest.raises(RuntimeError, match='^package PyICU'):
            simplify_unicode_chars(token, method=method)
    elif method == 'nonexistent':
        with pytest.raises(ValueError, match='`method` must be either "icu" or "ascii"'):
            simplify_unicode_chars(token, method=method)
    else:
        res = simplify_unicode_chars(token, method=method)
        assert isinstance(res, str)
        if method == 'icu' or (method == 'ascii' and ascii_encoding_errors == 'ignore'):
            assert len(res) <= len(token)

        if token in {'\u00C7', '\u0043\u0327', '\u0043\u0332'}:
            assert res == 'C'
        elif token == 'é':
            assert res == 'e'
        elif token == 'ῷ':
            if method == 'icu':
                assert res == 'ω'
            else:  # method == 'ascii'
                assert res == '' if ascii_encoding_errors == 'ignore' else '???'


@pytest.mark.parametrize('value, expected', [
    ('', ''),
    ('no tags', 'no tags'),
    ('<b>', ''),
    ('<b>x</b>', 'x'),
    ('<b>x &amp; y</b>', 'x & y'),
    ('<b>x &amp; <i>y</i> = &#9733;</b>', 'x & y = ★'),
    ('<b>x &amp; <i>y = &#9733;</b>', 'x & y = ★'),
])
def test_strip_tags(value, expected):
    assert strip_tags(value) == expected
