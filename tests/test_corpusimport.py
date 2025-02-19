"""
Tests for importing optional tmtoolkit.corpus module.

.. codeauthor:: Markus Konrad <post@mkonrad.net>
"""

from importlib.util import find_spec

import pytest


def test_import_corpus():
    if any(find_spec(pkg) is None for pkg in ('spacy', 'bidict', 'loky')):
        with pytest.raises(ImportError, match='^the required package'):
            from tmtoolkit import corpus
        with pytest.raises(ImportError, match='^the required package'):
            from tmtoolkit.corpus import Corpus
    else:
        from tmtoolkit import corpus
        from tmtoolkit.corpus import Corpus
        import spacy
        import bidict
        import loky
