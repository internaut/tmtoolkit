"""
Bag-of-Words (BoW) sub-package with modules for generating document-term-matrices (DTMs) and some common statistics for
the BoW model.

.. codeauthor:: Markus Konrad <post@mkonrad.net>
"""

from . import bow_stats, dtm
from ._naivebayes import NaiveBayesClassifier
