"""
Module for metrics.

.. codeauthor:: Markus Konrad <post@mkonrad.net>
"""


from __future__ import annotations

from functools import partial
from typing import Optional, Callable

import numpy as np


def pmi(x: np.ndarray, y: np.ndarray, xy: np.ndarray, n_total: Optional[int] = None, logfn: Callable = np.log,
        k: int = 1, normalize: bool = False) -> np.ndarray:
    """
    Calculate pointwise mutual information measure (PMI) either from probabilities p(x), p(y), p(x, y) given as `x`,
    `y`, `xy`, or from total counts `x`, `y`, `xy` and additionally `n_total`. Setting `k` > 1 gives PMI^k variants.
    Setting `normalized` to True gives normalized PMI (NPMI) as in [Bouma2009]_. See [RoleNadif2011]_ for a comparison
    of PMI variants.

    Probabilities should be such that ``p(x, y) <= min(p(x), p(y))``.

    :param x: probabilities p(x) or count of occurrences of x (interpreted as count if `n_total` is given)
    :param y: probabilities p(y) or count of occurrences of y (interpreted as count if `n_total` is given)
    :param xy: probabilities p(x, y) or count of occurrences of x *and* y (interpreted as count if `n_total` is given)
    :param n_total: if given, `x`, `y` and `xy` are interpreted as counts with `n_total` as size of the sample space
    :param logfn: logarithm function to use (default: ``np.log`` – natural logarithm)
    :param k: if `k` > 1, calculate PMI^k variant
    :param normalize: if True, normalize to range [-1, 1]; gives NPMI measure
    :return: array with same length as inputs containing (N)PMI measures for each input probability
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError('`k` must be a strictly positive integer')

    if k > 1 and normalize:
        raise ValueError('normalization is only implemented for standard PMI with `k=1`')

    if n_total is not None:
        if n_total < 1:
            raise ValueError('`n_total` must be strictly positive')
        x = x/n_total
        y = y/n_total
        xy = xy/n_total

    pmi_val = logfn(xy) - logfn(x) - logfn(y)

    if k > 1:
        return pmi_val - (1-k) * logfn(xy)
    else:
        if normalize:
            return pmi_val / -logfn(xy)
        else:
            return pmi_val


npmi = partial(pmi, k=1, normalize=True)
pmi2 = partial(pmi, k=2, normalize=False)
pmi3 = partial(pmi, k=3, normalize=False)


def ppmi(x: np.ndarray, y: np.ndarray, xy: np.ndarray, n_total: Optional[int] = None, logfn: Callable = np.log,
         k: int = 1, normalize: bool = False) -> np.ndarray:
    """
    Calculate positive pointwise mutual information measure (PPMI) as ``max(pmi(...), 0)``. Unless `normalize` is True,
    this results in a measure that is in range ``[0, +Inf]``. See :func:`pmi` for further information. See
    [JurafskyMartin2023]_, p. 117 for more on (positive) PMI.

    :param x: probabilities p(x) or count of occurrence of x (interpreted as count if `n_total` is given)
    :param y: probabilities p(y) or count of occurrence of y (interpreted as count if `n_total` is given)
    :param xy: probabilities p(x, y) or count of occurrence of x *and* y (interpreted as count if `n_total` is given)
    :param n_total: if given, `x`, `y` and `xy` are interpreted as counts with `n_total` as size of the sample space
    :param logfn: logarithm function to use (default: ``np.log`` – natural logarithm)
    :param k: if `k` > 1, calculate PMI^k variant
    :param normalize: if True, first normalize to range [-1, 1] and then apply ``max(npmi, 0)``
    :return: array with same length as inputs containing PPMI measures for each input probability
    """
    return np.maximum(pmi(x, y, xy, n_total=n_total, logfn=logfn, k=k, normalize=normalize), 0)


def simple_collocation_counts(x: Optional[np.ndarray], y: Optional[np.ndarray], xy: np.ndarray, n_total: Optional[int]):
    """
    "Statistic" function that can be used in :func:`~tmtoolkit.tokenseq.token_collocations` and will simply return the
    number of collocations between tokens *x* and *y* passed as `xy`. Mainly useful for debugging purposes.

    :param x: unused
    :param y: unused
    :param xy: counts for collocations of *x* and *y*
    :param n_total: total number of tokens (strictly positive)
    :return: simply returns `xy`
    """
    return xy.astype(float)
